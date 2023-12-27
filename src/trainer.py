from typing import Callable

import einops
import pytorch_lightning as pl
import timm.models as timm_models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch import Tensor
from torch.optim import AdamW, SGD
import utils
from models import models_dict


class ImgTrain(pl.LightningModule):
    def __init__(self,
                 num_classes,
                 learning_rate,
                 weight_decay,
                 maxepochs,
                 model_arch="resnet18",
                 lr_schedule=True,
                 milestones=[100, 150],
                 lr_decay=0.5,
                 ):
        super().__init__()

        if model_arch in models_dict:
            self.model = models_dict[model_arch](num_classes=num_classes)
        elif num_classes == 1000:
            self.model = models.__dict__[model_arch](pretrained=False, num_classes=num_classes)
        else:
            self.model = models.__dict__[model_arch](pretrained=False)
            self.model.fc = torch.nn.Linear(self.model.fc.weight.data.size(1), num_classes)

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_schedule = lr_schedule
        self.milestones = milestones
        self.lr_decay = lr_decay

        self.maxepochs = maxepochs
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        y = self.model(x)
        return y

    def training_step(self, batch, batch_idx):
        x, y = batch

        preds = self.model(x)
        loss = self.loss(preds, y)
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

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        preds = self.model(x)
        loss = F.cross_entropy(preds, y)
        topk = utils.accuracy(preds, y, topk=(1, 5))

        self.log("val/loss", loss,
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("val/acc_top1", topk[0],
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log("val/acc_top5", topk[1],
                 on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9,
                          weight_decay=self.weight_decay)
        return [optimizer]
