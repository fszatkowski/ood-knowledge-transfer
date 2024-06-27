import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch import Tensor
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from data_utils import cutmix, get_kornia_augs
from metrics import accuracy
from model_utils import init_model


class MainModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        assert cfg.mode in (
            "train",
            "distill",
            "eval",
        ), '"mode" must be one of ("train", "distill", "eval")'
        self.mode = cfg.mode

        if self.mode == "train":
            self.model = init_model(
                model_arch=cfg.model_arch,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                use_timm=cfg.use_timm,
            )
            if cfg.teacher_arch is None:
                raise ValueError(
                    'Teacher arch should be None with "train" mode. '
                    'Check if you do not want to use "distill"'
                )

            self.loss = nn.CrossEntropyLoss()
            self._train_step = self._standard_train_step

        elif self.mode == "distill":
            self.model = init_model(
                model_arch=cfg.model_arch,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                use_timm=cfg.use_timm,
            )
            self.teacher = init_model(
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
            self.model = init_model(
                model_arch=cfg.model_arch,
                num_classes=cfg.num_classes,
                pretrained=cfg.pretrained,
                use_timm=cfg.use_timm,
                from_checkpoint=cfg.model_ckpt,
            )
            self.loss = None

        # Training params
        self.maxepochs = cfg.maxepochs
        self.cfg = cfg

        # Augmentations
        self.use_kornia = cfg.use_kornia
        if self.use_kornia:
            self.train_augs = get_kornia_augs(cfg.train_dataset)[0]
            self.eval_augs = get_kornia_augs(cfg.test_dataset)[1]

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
            x = cutmix(x)

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
        top_k = accuracy(predictions, y, topk=(1, 2, 5))

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
        if self.cfg.optimizer.lower() == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
            )
        elif self.cfg.optimizer.lower() == "sgd":
            optimizer = SGD(
                self.model.parameters(),
                lr=self.cfg.learning_rate,
                weight_decay=self.cfg.weight_decay,
                momentum=self.cfg.momentum,
            )
        else:
            raise NotImplementedError()

        if self.cfg.scheduler.lower() == "step":
            scheduler = MultiStepLR(
                optimizer, milestones=self.cfg.milestones, gamma=self.cfg.lr_decay
            )
        elif self.cfg.scheduler.lower() == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.cfg.maxepochs, eta_min=1e-6
            )
        elif self.cfg.scheduler.lower() == "none":
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
