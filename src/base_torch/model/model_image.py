# -*- coding: utf-8 -*- #
"""Models for Image Data"""
import copy
import typing as tp
from abc import ABCMeta, abstractmethod
from pathlib import Path

import timm
import torch
from torch import nn

from .model_basic import MLP, Conv2dBNActiv

# # ---------- Meta Class for Image Backbone ----------- # #

class TimmBase(nn.Module, metaclass=ABCMeta):
    """
    Meta Class for using Image Backbone implemented by timm.

    Py'T'orch 'Im'age 'M'odels:
        repository: https://github.com/rwightman/pytorch-image-models
        docs: https://rwightman.github.io/pytorch-image-models/
    """

    def __init__(
        self,
        model_name     : str,
        pretrained     : bool=False,
        checkpoint_path: tp.Union[str, Path]=False,
        in_channels    : int=3,
        make_in_channels_same: bool=False,
        **backbone_kwargs: tp.Dict,
    ):
        """Initialize."""
        assert timm.is_model(model_name), "You can use only models in timm"
        self.backbone_name = model_name
        super().__init__()  # call nn.Module.__init__()
           
        if make_in_channels_same:
            base_model = timm.create_model(
                model_name, pretrained, checkpoint_path,
                num_classes=0, in_chans=1, **backbone_kwargs)
            layer_chain = base_model.default_cfg["first_conv"].split(".")
            l = base_model
            for l_name in layer_chain:
                if l_name.isdigit():
                    l = l[int(l_name)]
                else:
                    l = getattr(l, l_name)
            # # # duplicate weight
            w = l.weight.data.float()
            l.weight.data = w.repeat(1, in_channels, 1, 1) / in_channels
        else:
            base_model = timm.create_model(
                model_name, pretrained, checkpoint_path,
                num_classes=0, in_chans=in_channels, **backbone_kwargs)
        
        self.backbone = base_model
        print(f"[complete] backbone: {model_name}, pretrained: {pretrained}")
        self.head     = self.create_head()

    @abstractmethod
    def create_head(self):
        """Create head"""
        pass

    @abstractmethod
    def forward_head(self):
        """
        Forward head
        
        This method may include special reshaping of Tensors.
        """
        pass

    def forward(self, x):
        """Forward Data"""
        h = self.backbone(x)
        y = self.forward_head(h)
        return y


# # --------------- Basic Image Model --------------- # #

class BasicImageModel(TimmBase):
    
    def __init__(
        self,
        model_name     : str,
        dims_head      : tp.List[int],
        head_drop_rate : float=0.5,
        pretrained     : bool=False,
        checkpoint_path: tp.Union[str, Path]='',
        in_channels    : int=3,
        make_in_channels_same: bool=False,
        **backbone_kwargs
    ):
        """Initialize"""
        # # prepare backbone
        self.dims_head = dims_head
        self.drop_rate = head_drop_rate
        super().__init__(
            model_name, pretrained, checkpoint_path,
            in_channels, make_in_channels_same,
            **backbone_kwargs)

    def create_head(self):
        """Set MLP as head"""
        if self.dims_head[0] is None:
            self.dims_head[0] = self.backbone.num_features
        return MLP(self.dims_head, True, self.drop_rate)

    def forward_head(self, h):
        """Simply forward head layer"""
        return self.head(h)


class ABResNetD(TimmBase):
    """
    A imitation of Attention Branch Network
    
    This class uses ResNet-D variants by timm.
    """

    def __init__(
        self,
        model_name     : str,
        dims_head      : tp.List[int],
        num_classes_aux: int,
        pretrained     : bool=False,
        checkpoint_path: tp.Union[str, Path]='',
        in_channels          : int=3,
        make_in_channels_same: bool=False,
        **backbone_kwargs
    ):
        """Initialize"""
         # # prepare backbone
        model_list = [
            "resnet18d", "resnet26d", "resnet34d", "resnet50d",
            "resnet101d", "resnet152d", "resnet200d", "seresnet152d"]
        assert model_name in model_list, "You can use only resnet variants"
        super().__init__(
            model_name, pretrained, checkpoint_path,
            in_channels, make_in_channels_same,
            **backbone_kwargs)
        base_model = self.backbone
        del self.backbone
        
        self.dims_head   = dims_head
        self.in_features = base_model.num_features
        print(f"{model_name}: {self.in_features}")

        # # feature extractor
        self.extractor = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.act1, base_model.maxpool,
            base_model.layer1, base_model.layer2, base_model.layer3)
        
        # # attention branch
        in_channels = self.in_features

        attn_layer4 = copy.deepcopy(base_model.layer4)
        attn_layer4[0].downsample[0] = nn.Identity()
        attn_layer4[0].conv1.stride = (1, 1)
        self.attn_neck = nn.Sequential(
            attn_layer4, nn.BatchNorm2d(in_channels),
            Conv2dBNActiv(in_channels, num_classes_aux, 1))

        self.attn_head = Conv2dBNActiv(
            num_classes_aux, 1, 3, padding=1, activ="sigmoid")

        self.aux_head = nn.Sequential(
            nn.Conv2d(num_classes_aux, num_classes_aux, 1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1))
        
        # # perception branch
        self.perc_neck = nn.Sequential(
           base_model.layer4,
           nn.AdaptiveAvgPool2d(1), nn.Flatten(1))
        

    def create_head(self):
        """Set MLP as head"""
        if self.dims_head[0] is None:
            self.dims_head[0] = self.in_features
        return MLP(self.dims_head, True, 0.5)

    def forward_head(self, h):
        """"""
        perc_h = h * self.attn
        perc_h = perc_h + h
        h      = self.perc_neck(perc_h)
        return self.head(h)

    def forward(self, x):
        """Forward"""
        extr_h = self.extractor(x)

        attn_h = self.attn_neck(extr_h)
        # bs, cs, ys, xs = attn_h.shape
        self.attn = self.attn_head(attn_h)
        y = self.forward_head(extr_h)

        if not self.training:
            return y

        y_aux = self.aux_head(attn_h)
        # return y, y_aux, [self.att, extr_h, perc_h]
        return y, y_aux

