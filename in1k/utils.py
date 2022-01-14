from PIL import Image, ImageFilter, ImageOps
import math
import random
import numpy as np

import torch
import torchvision.transforms.functional as tf
import pytorch_lightning as pl
        
class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""
    def __call__(self, x):
        return ImageOps.solarize(x)
        
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class CheckpointEveryEpoch(pl.Callback):
    def __init__(self, every, save_path):
        self.every = every
        self.save_path = save_path

    def on_epoch_end(self, trainer: pl.Trainer, _):
        """ Check if we should save a checkpoint after every train epoch """
        epoch = trainer.current_epoch
        if epoch % self.every == 0:
            ckpt_path = f"{self.save_path}/ckpt_{epoch}.ckpt"
            trainer.save_checkpoint(ckpt_path)
