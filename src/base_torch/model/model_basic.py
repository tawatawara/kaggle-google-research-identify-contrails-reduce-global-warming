# -*- coding: utf-8 -*- #
"""Basic Model Layers"""
import typing as tp
from pathlib import Path

import torch
from torch import nn


# # ------------------- Basic FC Layer ------------------- # # 

def get_activation(activ_name: str="relu"):
    """"""
    act_dict = {
        "relu": nn.ReLU(inplace=True),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "identity": nn.Identity()}
    if activ_name in act_dict:
        return act_dict[activ_name]
    else:
        raise NotImplementedError


class LBAD(nn.Module):
    """Linear (-> BN) -> Activation (-> Dropout)"""
    
    def __init__(
        self, in_features: int, out_features: int, drop_rate: float=0.0,
        use_bn: bool=False, use_wn: bool=False, activ: str="relu"
    ):
        """"""
        super().__init__()
        layers = [nn.Linear(in_features, out_features)]
        if use_wn:
            layers[0] = nn.utils.weight_norm(layers[0])
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
        layers.append(get_activation(activ))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)
    
    
class BDLA(nn.Module):
    """(BN -> Dropout ->) Linear -> Activation"""
    
    def __init__(
        self, in_features: int, out_features: int, drop_rate: float=0.0,
        use_bn: bool=False, use_wn: bool=False, activ: str="relu"
    ):
        """"""
        super().__init__()
        layers = []
        if use_bn:
            layers.append(nn.BatchNorm1d(in_features))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        layers.append(nn.Linear(in_features, out_features))
        if use_wn:
            layers[-1] = nn.utils.weight_norm(layers[-1])
        layers.append(get_activation(activ))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)
    

class LABD(nn.Module):
    """Linear -> Activation (-> BN -> Dropout) """
    
    def __init__(
        self, in_features: int, out_features: int, drop_rate: float=0.0,
        use_bn: bool=False, use_wn: bool=False, activ: str="relu"
    ):
        """"""
        super().__init__()
        layers = [nn.Linear(in_features, out_features), get_activation(activ)]
        if use_wn:
            layers[0] = nn.utils.weight_norm(layers[0])
        if use_bn:
            layers.append(nn.BatchNorm1d(out_features))
        if drop_rate > 0:
            layers.append(nn.Dropout(drop_rate))
        
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)


class MLP(nn.Module):
    """Stacked Dense layers"""
    
    def __init__(
        self, n_features_list: tp.List[int], use_tail_as_out: bool=False,
        drop_rate: float=0.0, use_bn: bool=False, use_wn: bool=False,
        activ:str="relu", block_type: str="LBAD"
    ):
        """"""
        super().__init__()
        n_layers = len(n_features_list) - 1
        block_class = {
            "LBAD": LBAD, "BDLA": BDLA, "LABD": LABD}[block_type]
        layers = []
        for i in range(n_layers):
            in_feats, out_feats = n_features_list[i: i + 2]
            if i == n_layers - 1 and use_tail_as_out:
                if block_type in ["BDLA"]:
                    layer = block_class(
                        in_feats, out_feats, drop_rate, use_bn,  use_wn, "identity")
                else:
                    layer = nn.Linear(in_feats, out_feats)
                    if use_wn:
                        layer = nn.utils.weight_norm(layer)
            else:
                layer = block_class(
                    in_feats, out_feats, drop_rate, use_bn,  use_wn, activ)
            layers.append(layer)
                
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """"""
        return self.layers(x)


# # ------------------- Basic Conv Layer ------------------- # #

class Conv2dBNActiv(nn.Module):
    """Conv2d -> (BN ->) -> Activation"""
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: int=1, padding: int=0,
        bias: bool=False, use_bn: bool=True, activ: str="relu"
    ):
        """"""
        super(Conv2dBNActiv, self).__init__()
        layers = []
        layers.append(nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward"""
        return self.layers(x)


class Conv1dBNActiv(nn.Module):
    """Conv1d -> (BN ->) -> Activation"""
    
    def __init__(
        self, in_channels: int, out_channels: int,
        kernel_size: int, stride: int=1, padding: int=0,
        bias: bool=False, use_bn: bool=True, activ: str="relu"
    ):
        """"""
        super(Conv1dBNActiv, self).__init__()
        layers = []
        layers.append(nn.Conv1d(
            in_channels, out_channels,
            kernel_size, stride, padding, bias=bias))
        if use_bn:
            layers.append(nn.BatchNorm1d(out_channels))
            
        layers.append(get_activation(activ))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """Forward"""
        return self.layers(x)
