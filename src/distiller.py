from typing import Callable

import einops
import pytorch_lightning as pl
import timm.models as timm_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torch.optim import AdamW
import utils
from models import models_dict


class ImgDistill(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 teacher_ckpt,
                 learning_rate=0.01,
                 weight_decay=0.05,
                 temperature=8,
                 maxepochs=200,
                 student_arch="resnet18",
                 teacher_arch="resnet50",
                 lr_schedule=True,
                 use_timm=False,
                 milestones=[100, 150],
                 lr_decay=0.5,
                 n_rvectors=[12, 13, 14, 15],
                 n_projections=20,
                 ):
        super().__init__()

        if teacher_arch in models_dict:
            self.teacher = models_dict[teacher_arch](num_classes=num_classes)
        elif use_timm:
            self.teacher = timm_models.__dict__[teacher_arch](pretrained=num_classes == 1000, num_classes=num_classes)
        else:
            self.teacher = models.__dict__[teacher_arch](pretrained=num_classes == 1000, num_classes=num_classes)
        if teacher_ckpt != "":
            state_dict = torch.load(teacher_ckpt, map_location="cpu")["state_dict"]
            # TODO fix for CIFAR 100 archs - is it even necessary if we match train and distill code?
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            self.teacher.load_state_dict(state_dict)
        self.teacher_features = None
        self.teacher.fc.register_forward_hook(self.teacher_forward_hook())

        if teacher_arch in models_dict:
            self.student = models_dict[student_arch](num_classes=num_classes)
        elif num_classes == 1000:
            self.student = models.__dict__[student_arch](pretrained=False, num_classes=num_classes)
        else:
            self.student = models.__dict__[student_arch](pretrained=False)
            self.student.fc = torch.nn.Linear(self.student.fc.weight.data.size(1), num_classes)

        self.n_rvectors = n_rvectors
        self.n_projections = n_projections
        self.projections: dict[int, Tensor] = {}
        self.multiply_matrix: dict[int, Tensor] = {}
        self.bucket_cover_matrix_per_epoch: dict[int, Tensor] = {}
        self.bucket_cover_matrix_total: dict[int, Tensor] = {}

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.temperature = temperature
        self.lr_schedule = lr_schedule
        self.milestones = milestones
        self.lr_decay = lr_decay

        self.maxepochs = maxepochs
        self.with_cutmix = True
        self.loss = nn.KLDivLoss(reduction="batchmean")
        self.teacher.eval()

        for param in self.teacher.parameters():
            param.requires_grad = False

    def teacher_forward_hook(self) -> Callable:
        def fn(_, __, output):
            self.teacher_features = output

        return fn

    def kd_loss_fn(self, outputs, teacher_outputs):
        kd_loss = self.loss(F.log_softmax(outputs / self.temperature, dim=1),
                            F.softmax(teacher_outputs / self.temperature, dim=1))
        return kd_loss

    def forward(self, x):
        y = self.student(x)
        return y

    def on_train_epoch_start(self):
        self.teacher.eval()
        self.bucket_cover_matrix_per_epoch = {}

    def on_train_epoch_end(self, **kwargs):
        for n_vectors in self.n_rvectors:
            buckets_per_epoch = self.bucket_cover_matrix_per_epoch[n_vectors].sum(dim=1) / 2 ** n_vectors
            buckets_per_epoch = buckets_per_epoch.mean()
            self.log(f"bin_cov_{n_vectors}/epoch", 100 * float(buckets_per_epoch), logger=True)
            buckets_total = self.bucket_cover_matrix_total[n_vectors].sum(dim=1) / 2 ** n_vectors
            buckets_total = buckets_total.mean()
            self.log(f"bin_cov_{n_vectors}/total", 100 * float(buckets_total), logger=True)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.with_cutmix:
            x = utils.cutmixed(x)

        with torch.no_grad():
            teacher_predictions = self.teacher(x)
        student_predictions = self.student(x)
        loss = self.kd_loss_fn(student_predictions, teacher_predictions)
        self.log("train/loss", loss, on_step=True, on_epoch=False,
                 prog_bar=True, logger=True)
        if self.lr_schedule:
            if self.trainer.is_last_batch:
                lr = self.learning_rate
                for milestone in self.milestones:
                    lr *= self.lr_decay if self.current_epoch >= milestone else 1.
                print(f"LR={lr}")
                print()
                for param_group in self.optimizers().param_groups:
                    param_group["lr"] = lr

        self.update_buckets()

        return loss

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, batch_idx, 'test')

    def _evaluation_step(self, batch, batch_idx, key='val'):
        x, y = batch

        student_predictions = self.student(x)
        loss = F.cross_entropy(student_predictions, y)
        topk = utils.accuracy(student_predictions, y, topk=(1, 5))

        self.log(f"{key}/loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log(f"{key}/acc_top1", topk[0],
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log(f"{key}/acc_top5", topk[1],
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.student.parameters(), lr=self.learning_rate,
                          weight_decay=self.weight_decay)
        return [optimizer]

    def update_buckets(self):
        for n_vectors in self.n_rvectors:
            if n_vectors not in self.bucket_cover_matrix_per_epoch:
                self.bucket_cover_matrix_per_epoch[n_vectors] = torch.zeros((self.n_projections, 2 ** n_vectors),
                                                                            dtype=torch.bool).to(
                    self.teacher_features.device, non_blocking=True)
            if n_vectors not in self.bucket_cover_matrix_total:
                self.bucket_cover_matrix_total[n_vectors] = torch.zeros((self.n_projections, 2 ** n_vectors),
                                                                        dtype=torch.bool).to(
                    self.teacher_features.device, non_blocking=True)
            if n_vectors not in self.projections:
                projections = [
                    torch.randn(self.teacher.fc.out_features, n_vectors).type_as(self.teacher_features) for _ in
                    range(self.n_projections)]
                self.projections[n_vectors] = torch.stack(projections, dim=0).to(self.teacher_features.device,
                                                                                 non_blocking=True)
            if n_vectors not in self.multiply_matrix:
                self.multiply_matrix = {}
                mask = torch.ones((n_vectors,)) * 2
                exponential = torch.arange(n_vectors)
                self.multiply_matrix[n_vectors] = torch.pow(mask, exponential).type(
                    torch.int64).unsqueeze(0).unsqueeze(0).to(self.teacher_features.device, non_blocking=True)

            teacher_features = einops.repeat(self.teacher_features, 'b f -> n b f',
                                             n=self.projections[n_vectors].shape[0])
            result: Tensor = torch.bmm(teacher_features, self.projections[n_vectors])  # [n, b, v]
            binary_hash = torch.gt(result, 0)  # [n, b, v]
            bin_ids = (binary_hash * self.multiply_matrix[n_vectors]).sum(2).type(torch.int64)  # [n, b]
            covered_bins = torch.nn.functional.one_hot(bin_ids, num_classes=2 ** n_vectors).sum(dim=1).to(
                self.teacher_features.device, non_blocking=True)  # [n, 2**v]
            covered_bins = covered_bins > 0  # convert to bool [n, 2**v]
            self.bucket_cover_matrix_per_epoch[n_vectors] += covered_bins
            self.bucket_cover_matrix_total[n_vectors] += covered_bins
