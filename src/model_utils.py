from typing import Optional

import timm.models as timm_models
import torch
import torchvision.models as tv_models

from models import models_dict


def init_model(
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
