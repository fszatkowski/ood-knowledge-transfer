from typing import Optional

import pytorch_lightning as pl
import timm.models as timm_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

import utils
from data_utils import get_kornia_augs
from models import models_dict


class MainModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        assert cfg.mode in (
            "train",
            "distill",
            "eval",
        ), '"mode" must be one of ("train", "distill", "eval")'
        self.mode = cfg.mode
        self.model = self._init_model(
            model_arch=cfg.model_arch,
            num_classes=cfg.num_classes,
            pretrained=cfg.pretrained,
            use_timm=cfg.use_timm,
        )
        if self.mode == "train":
            if cfg.teacher_arch is not None:
                raise ValueError(
                    'Teacher arch should be None with "train" mode. '
                    'Check if you do not want to use "distill"'
                )

            self.loss = nn.CrossEntropyLoss()
            self._train_step = self._standard_train_step

        elif self.mode == "distill":
            self.teacher = self._init_model(
                model_arch=cfg.teacher_arch,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                use_timm=cfg.use_timm,
                from_checkpoint=cfg.teacher_ckpt,
            )
            for param in self.teacher.parameters():
                param.requires_grad = False
            self.teacher.eval()
            self.cutmix = cfg.cutmix  # Only applies to distillation
            self.loss = KDLoss(temperature=cfg.temperature)
            self._train_step = self._distillation_train_step

        elif self.mode == "eval":
            self.loss = None

        # Training params
        self.maxepochs = cfg.maxepochs
        self.learning_rate = cfg.learning_rate
        self.weight_decay = cfg.weight_decay
        self.momentum = cfg.momentum
        self.optimizer = cfg.optimizer
        self.scheduler = cfg.scheduler
        self.milestones = cfg.milestones
        self.lr_decay = cfg.lr_decay

        # Augmentations
        self.use_kornia = cfg.use_kornia
        if self.use_kornia:
            self.train_augs = get_kornia_augs(cfg.train_dataset)[0]
            self.eval_augs = get_kornia_augs(cfg.test_dataset)[1]

    @staticmethod
    def _init_model(
        model_arch: str,
        num_classes: int = None,
        pretrained: bool = False,
        use_timm: bool = False,
        from_checkpoint: Optional[str] = None,
    ):
        if model_arch in models_dict:
            # Custom models
            model = models_dict[model_arch](num_classes=num_classes)
        elif use_timm:
            # TIMM models
            model = timm_models.__dict__[model_arch](
                pretrained=pretrained, num_classes=num_classes
            )
        else:
            # torchvision models
            model = tv_models.__dict__[model_arch](
                pretrained=pretrained, num_classes=num_classes
            )

        if from_checkpoint is not None:
            # Lightning adds prefixes that must be removed to load the model weights
            state_dict = torch.load(from_checkpoint, map_location="cpu")["state_dict"]
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("teacher.", ""): v for k, v in state_dict.items()}
            state_dict = {
                k: v for k, v in state_dict.items() if k in model.state_dict().keys()
            }
            assert len(state_dict) == len(model.state_dict().keys())
            model.load_state_dict(state_dict)
        return model

    def forward(self, x: Tensor) -> Tensor:
        y = self.model(x)
        return y

    def on_train_start(self):
        if self.mode == "distill":
            # TODO maybe add one-time evaluation on the teacher model for reference
            pass

    def on_train_epoch_start(self):
        self.model.train()
        if self.mode == "distill":
            # TODO check if teacher adaptation does not help :)
            self.teacher.eval()

    def training_step(self, batch, batch_idx):
        return self._train_step(batch, batch_idx)

    def _standard_train_step(self, batch, batch_idx):
        x, y = batch
        if self.use_kornia:
            x = self.train_augs(x)

        predictions = self.model(x)
        loss = self.loss(predictions, y)
        self.log(
            "train/loss",
            float(loss),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def _distillation_train_step(self, batch, batch_idx):
        x, y = batch
        if self.use_kornia:
            x = self.train_augs(x)
        if self.cutmix:
            x = utils.cutmixed(x)

        with torch.no_grad():
            teacher_predictions = self.teacher(x)
        student_predictions = self.model(x)
        loss = self.loss(student_predictions, teacher_predictions)
        self.log(
            "train/loss",
            float(loss),
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(self.model, batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._evaluation_step(self.model, batch, batch_idx, "test")

        # TODO add additional metrics (e.g. latent coverage)

    @torch.inference_mode()
    def _evaluation_step(self, model, batch, batch_idx, key="val"):
        x, y = batch

        if self.use_kornia:
            x = self.eval_augs(x)

        predictions = model(x)
        loss = F.cross_entropy(predictions, y)
        top_k = utils.accuracy(predictions, y, topk=(1, 2, 5))

        self.log(
            f"{key}/loss",
            float(loss),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{key}/acc_top1",
            top_k[0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{key}/acc_top2",
            top_k[1],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            f"{key}/acc_top5",
            top_k[2],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def configure_optimizers(self):
        if self.optimizer.lower() == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer.lower() == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum,
            )
        else:
            raise NotImplementedError()

        if self.scheduler.lower() == "step":
            scheduler = MultiStepLR(
                optimizer, milestones=self.milestones, gamma=self.lr_decay
            )
        elif self.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.maxepochs, eta_min=1e-6)
        elif self.scheduler.lower() == "none":
            scheduler = None
        else:
            raise NotImplementedError()

        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


class KDLoss(nn.Module):
    def __init__(self, temperature: float):
        super(KDLoss, self).__init__()
        self.temperature = temperature
        self.loss = nn.KLDivLoss(reduction="batchmean")

    def forward(self, outputs: Tensor, teacher_outputs: Tensor) -> Tensor:
        return self.loss(
            F.log_softmax(outputs / self.temperature, dim=1),
            F.softmax(teacher_outputs / self.temperature, dim=1),
        )
