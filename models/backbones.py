import einops
import torch
import torch.nn as nn
import dataclasses
import torchvision

from torchvision.transforms import Resize
from typing import Optional, Dict, Callable
from torchvision.models import (
    resnet18,
    resnet34,
    resnet50,
    resnet101,
    resnet152,
    vit_b_16,
    vit_b_32,
    vit_l_16,
    vit_l_32,
    vit_h_14,
)


@dataclasses.dataclass
class TorchvisionModelMetadata:
    model: Callable[[str], nn.Module]
    emb_dims_last_layer: int
    expected_image_size: Optional[tuple]


_torchvision_resnet_model_metadata: Dict[str, TorchvisionModelMetadata] = {
    "resnet18": TorchvisionModelMetadata(
        model=resnet18,
        emb_dims_last_layer=512,
        expected_image_size=None,
    ),
    "resnet34": TorchvisionModelMetadata(
        model=resnet34,
        emb_dims_last_layer=512,
        expected_image_size=(224, 224),
    ),
    "resnet50": TorchvisionModelMetadata(
        model=resnet50,
        emb_dims_last_layer=2048,
        expected_image_size=None,
    ),
    "resnet101": TorchvisionModelMetadata(
        model=resnet101,
        emb_dims_last_layer=2048,
        expected_image_size=None,
    ),
    "resnet152": TorchvisionModelMetadata(
        model=resnet152,
        emb_dims_last_layer=2048,
        expected_image_size=None,
    ),
}

_torchvision_vit_model_metadata: Dict[str, TorchvisionModelMetadata] = {
    "vit_b_16": TorchvisionModelMetadata(
        model=vit_b_16,
        emb_dims_last_layer=768,
        expected_image_size=(224, 224),
    ),
    "vit_b_32": TorchvisionModelMetadata(
        model=vit_b_32,
        emb_dims_last_layer=768,
        expected_image_size=(224, 224),
    ),
    "vit_l_16": TorchvisionModelMetadata(
        model=vit_l_16,
        emb_dims_last_layer=1024,
        expected_image_size=(224, 224),
    ),
    "vit_l_32": TorchvisionModelMetadata(
        model=vit_l_32,
        emb_dims_last_layer=1024,
        expected_image_size=(224, 224),
    ),
    "vit_h_14": TorchvisionModelMetadata(
        model=vit_h_14,
        emb_dims_last_layer=1280,
        expected_image_size=(518, 518),
    ),
}


def _get_module_dict(model: nn.Module) -> Dict[str, nn.Module]:
    """
    Returns a dictionary of all modules in the model.
    """
    module_dict = {}
    for name, module in model.named_modules():
        module_dict[name] = module
    return module_dict


class ResBackbone(nn.Module):
    def __init__(
        self,
        emb_dim: int,
        model_name: str,
        weights: str,
    ):
        super(ResBackbone, self).__init__()
        self.model_name = model_name
        if model_name not in _torchvision_resnet_model_metadata:
            raise ValueError(f"model_name {model_name} not in torchvision models")

        self.model = _torchvision_resnet_model_metadata[model_name].model(
            weights=weights
        )

        self.module_dict = _get_module_dict(self.model)

        # Each layers are borrowed from the implemenration of
        # `class torchvision.resnet.ResNet._forward_impl(self, x: Tensor) -> Tensor`
        self.conv1 = self.module_dict["conv1"]
        self.bn1 = self.module_dict["bn1"]
        self.relu = self.module_dict["relu"]
        self.maxpool = self.module_dict["maxpool"]

        self.layer1 = self.module_dict["layer1"]
        self.layer2 = self.module_dict["layer2"]
        self.layer3 = self.module_dict["layer3"]
        self.layer4 = self.module_dict["layer4"]

        self.final_layer = nn.Linear(
            _torchvision_resnet_model_metadata[model_name].emb_dims_last_layer, emb_dim
        )

    def forward(self, x):
        """
        Createa an embedding of given image `x` using the backbone model.
        Args:
        x (torch.Tensor): The input tensor of shape [B, H, W, C].
        """
        x = einops.rearrange(x, "b h w c -> b c h w")

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = einops.rearrange(x, "b c h w -> b (h w) c")
        x = self.final_layer(x)
        return x


class ViTBackbone(nn.Module):
    def __init__(self, emb_dim: int, model_name: str, weights: str):
        super(ViTBackbone, self).__init__()
        self.model_name = model_name
        if model_name not in _torchvision_vit_model_metadata:
            raise ValueError(f"model_name {model_name} not in torchvision models")
        self.model: torchvision.models.VisionTransformer = (
            _torchvision_vit_model_metadata[model_name].model(weights=weights)
        )

        self.resizer = Resize(
            _torchvision_vit_model_metadata[model_name].expected_image_size
        )

        self.final_layer = nn.Linear(
            _torchvision_vit_model_metadata[model_name].emb_dims_last_layer, emb_dim
        )

    def forward(self, x):
        """
        Createa an embedding of given image `x` using the backbone model.
        Args:
        x (torch.Tensor): The input tensor of shape [B, H, W, C].
        """
        x = einops.rearrange(x, "b h w c -> b c h w")
        x = self.resizer(x)

        # the implementaion of this code is borrowed from `torchvision.models.vision_transformer`
        # `class torchvision.models.vision_transformer.VisionTransformer.forward(self, x: Tensor) -> Tensor`
        x = self.model._process_input(x)
        n = x.shape[0]

        batch_calss_token = self.model.class_token.expand(n, -1, -1)

        x = torch.cat((batch_calss_token, x), dim=1)

        x = self.model.encoder(x)

        x = self.final_layer(x)

        return x


def _to_backbone_weights_option(pretrained: bool = True):
    if pretrained:
        return "DEFAULT"
    else:
        return None


def create_backbone(model_name: str, emb_dim: int, pretrained: bool = True):
    if "res" in model_name:
        return ResBackbone(
            emb_dim=emb_dim,
            model_name=model_name,
            weights=_to_backbone_weights_option(pretrained),
        )
    elif "vit" in model_name:
        return ViTBackbone(
            emb_dim=emb_dim,
            model_name=model_name,
            weights=_to_backbone_weights_option(pretrained),
        )
    else:
        raise ValueError(f"model_name {model_name} not in torchvision models")


def create_resnet18_backbone(emb_dim: int, pretrained: bool = True):
    return ResBackbone(
        emb_dim=emb_dim,
        model_name="resnet18",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_resnet34_backbone(emb_dim: int, pretrained: bool = True):
    return ResBackbone(
        emb_dim=emb_dim,
        model_name="resnet34",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_resnet50_backbone(emb_dim: int, pretrained: bool = True):
    return ResBackbone(
        emb_dim=emb_dim,
        model_name="resnet50",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_resnet101_backbone(emb_dim: int, pretrained: bool = True):
    return ResBackbone(
        emb_dim=emb_dim,
        model_name="resnet101",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_resnet152_backbone(emb_dim: int, pretrained: bool = True):
    return ResBackbone(
        emb_dim=emb_dim,
        model_name="resnet152",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_vit_b_16_backbone(emb_dim: int, pretrained: bool = True):
    return ViTBackbone(
        model_name="vit_b_16",
        emb_dim=emb_dim,
        weights=_to_backbone_weights_option(pretrained),
    )


def create_vit_b_32_backbone(emb_dim: int, pretrained: bool = True):
    return ViTBackbone(
        emb_dim=emb_dim,
        model_name="vit_b_32",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_vit_l_16_backbone(emb_dim: int, pretrained: bool = True):
    return ViTBackbone(
        emb_dim=emb_dim,
        model_name="vit_l_16",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_vit_l_32_backbone(emb_dim: int, pretrained: bool = True):
    return ViTBackbone(
        model_name="vit_l_32",
        weights=_to_backbone_weights_option(pretrained),
    )


def create_vit_h_14_backbone(emb_dim: int, pretrained: bool = True):
    return ViTBackbone(
        emb_dim=emb_dim,
        model_name="vit_h_14",
        weights=_to_backbone_weights_option(pretrained),
    )
