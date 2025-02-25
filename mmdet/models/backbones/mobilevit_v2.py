"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""
import warnings
from typing import Dict, Optional, Sequence, Tuple, Union
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models.utils.transformer2 import LinearAttnFFN
from ..builder import BACKBONES
from ..utils import InvertedResidual, make_divisible
from torch.nn.modules.batchnorm import _BatchNorm
from ..plugins import DropBlock

class MobileViTBlockv2(nn.Module):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(
        self,
        in_channels: int,
        attn_unit_dim: int,
        ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
        n_attn_blocks: Optional[int] = 2,
        attn_dropout: Optional[float] = 0.0,
        dropout: Optional[float] = 0.0,
        ffn_dropout: Optional[float] = 0.0,
        patch_h: Optional[int] = 8,
        patch_w: Optional[int] = 8,
        conv_ksize: Optional[int] = 3,
        dilation: Optional[int] = 1,
        attn_norm_layer: Optional[str] = "layer_norm_2d",
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        cnn_out_dim = attn_unit_dim

        conv_3x3_in = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            padding=int((conv_ksize - 1) / 2),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'),
            groups=in_channels)

        conv_1x1_in = ConvModule(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_attn_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
            attn_norm_layer=attn_norm_layer,
        )

        self.conv_proj = ConvModule(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU')
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_attn_blocks
        self.conv_ksize = conv_ksize

    def _compute_unfolding_weights(self) -> Tensor:
        # [P_h * P_w, P_h * P_w]
        weights = torch.eye(self.patch_h * self.patch_w, dtype=torch.float)
        # [P_h * P_w, P_h * P_w] --> [P_h * P_w, 1, P_h, P_w]
        weights = weights.reshape(
            (self.patch_h * self.patch_w, 1, self.patch_h, self.patch_w)
        )
        # [P_h * P_w, 1, P_h, P_w] --> [P_h * P_w * C, 1, P_h, P_w]
        weights = weights.repeat(self.cnn_out_dim, 1, 1, 1)
        return weights


    def _build_attn_layer(
        self,
        d_model: int,
        ffn_mult: Union[Sequence, int, float],
        n_layers: int,
        attn_dropout: float,
        dropout: float,
        ffn_dropout: float,
        attn_norm_layer: str,
        *args,
        **kwargs
    ) -> Tuple[nn.Module, int]:

        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        # ensure that dims are multiple of 16
        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
                norm_layer=attn_norm_layer,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(nn.GroupNorm(1,d_model))


        return nn.Sequential(*global_rep), d_model

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map

    def resize_input_if_needed(self, x):
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)

        # convert feature map to patches
        patches, output_size = self.unfolding_pytorch(fm)

        # learn global representations
        for global_layer in self.global_rep:
            patches = global_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding_pytorch(patches=patches, output_size=output_size)

        fm = self.conv_proj(fm)

        return fm

@BACKBONES.register_module()
class MobileViTv2(BaseModule):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ architecture
    """

    # Parameters to build layers. 4 parameters are needed to construct a
    # mv2_arch_settings layer, from left to right: expand_ratio, channel, num_blocks, stride.
    # mobilevitv2_arch_settings layer, from left to right: expand_ratio, out_channels, attn_unit_dim, ffn_multiplier, attn_blocks, patch_h, patch_w, stride,

    # layer 1 2
    mv2_arch_settings = [[2, 64, 1, 1], [2, 128, 2, 2]]
    # layer 3 4 5
    mobilevitv2_arch_settings=[[2, 256, 128, 2, 2, 2, 2, 2],
                             [2, 384, 192, 2, 4, 2, 2, 2],
                             [2, 512, 256, 2, 3, 2, 2, 2]]

    def __init__(self,
                 out_indices=(3, 4),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None,
                 dropout=0.1,
                 ffn_dropout=0.0,
                 attn_dropout=0.1,
                 last_layer_exp_factor=4,
                 use_dropoutblock=False,
                 ):

        super(MobileViTv2, self).__init__(init_cfg)
        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        image_channels = 3
        out_channels = 16
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.dropout = dropout
        self.ffn_dropout = ffn_dropout
        self.attn_dropout = attn_dropout
        self.last_layer_exp_factor=last_layer_exp_factor
        self.use_dropoutblock = use_dropoutblock

        if self.use_dropoutblock:
            dropblock = DropBlock(drop_prob=0.9, block_size=7)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.in_channels = out_channels

        self.layers = []

        for i, mv2_layer_cfg in enumerate(self.mv2_arch_settings):
            expand_ratio, out_channels, num_blocks, stride = mv2_layer_cfg

            inverted_res_layer, out_channels = self.make_mobilenet_layer(
                input_channel=self.in_channels,
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)
            self.in_channels = out_channels

        for i, mobilevitv2_layer_cfg in enumerate(self.mobilevitv2_arch_settings):
            expand_ratio, out_channels, attn_unit_dim, ffn_multiplier, \
            attn_blocks,patch_h, patch_w, stride= mobilevitv2_layer_cfg

            mit_layer, out_channels = self.make_mit_layer(
                input_channel=self.in_channels,
                out_channels=out_channels,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                attn_blocks=attn_blocks,
                patch_h=patch_h,
                patch_w=patch_w,
                stride=stride,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 3}'
            self.add_module(layer_name, mit_layer)
            self.layers.append(layer_name)
            self.in_channels = out_channels

            if i ==1 and self.use_dropoutblock:
                layer_name = 'drop1'
                self.add_module(layer_name, dropblock)
                self.layers.append(layer_name)
                self.out_indices[-1] = self.out_indices[-1] + 1


    def make_mobilenet_layer(self, input_channel: int, out_channels, num_blocks, stride, expand_ratio) -> Tuple[nn.Sequential, int]:
        block = []
        for i in range(num_blocks):
            stride = stride if i == 0 else 1
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=out_channels,
                mid_channels=make_divisible(int(round(input_channel * expand_ratio)), 8),
                stride=stride,
                with_expand_conv=expand_ratio != 1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp
            )
            block.append(layer)
            input_channel = out_channels

        return nn.Sequential(*block), input_channel


    def make_mit_layer(self, input_channel, out_channels, attn_unit_dim, ffn_multiplier, attn_blocks,
                           patch_h, patch_w, stride, expand_ratio) -> [nn.Sequential, int]:
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=out_channels,
                mid_channels=make_divisible(int(round(input_channel * expand_ratio)), 8),
                stride=stride,
                with_expand_conv=expand_ratio != 1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
                with_cp=self.with_cp
            )

            block.append(layer)
            input_channel = out_channels

        attn_unit_dim = attn_unit_dim
        ffn_multiplier = ffn_multiplier

        block.append(
            MobileViTBlockv2(
                in_channels=input_channel,
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_attn_blocks=attn_blocks,
                patch_h=patch_h,
                patch_w=patch_w,
                dropout=self.dropout,
                ffn_dropout=self.ffn_dropout,
                attn_dropout=self.attn_dropout,
                conv_ksize=3,
            ))
        input_channel = out_channels

        return nn.Sequential(*block), input_channel

    def forward(self, x):
        """Forward function."""
        x = self.conv1(x)
        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        # 调试
        # for i in range(len(outs)):
        #     print(f'outs{i}.shape:', outs[i].shape)
        # outs0.shape: torch.Size([8, 128, 80, 80])
        # outs1.shape: torch.Size([8, 256, 40, 40])
        # outs2.shape: torch.Size([8, 384, 20, 20])
        # outs3.shape: torch.Size([8, 512, 10, 10])
        # assert 1==2
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False
