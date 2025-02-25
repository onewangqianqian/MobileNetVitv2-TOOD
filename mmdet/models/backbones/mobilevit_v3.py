"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""
import warnings
from typing import Optional, Tuple, Union, Dict
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmdet.models.utils.transformer2 import  TransformerEncoder
from ..builder import BACKBONES
from ..utils import InvertedResidual, make_divisible
from torch.nn.modules.batchnorm import _BatchNorm


class MobileViTv3Block(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
        self,
        in_channels: int,
        transformer_dim: int,
        ffn_dim: int,
        n_transformer_blocks: int = 2,
        head_dim: int = 32,
        attn_dropout: float = 0.0,
        dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        patch_h: int = 8,
        patch_w: int = 8,
        conv_ksize: Optional[int] = 3,
        dilation = 1,
        var_ffn = False,
        no_fusion = False,
        *args,
        **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvModule(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            padding=int((conv_ksize - 1) / 2),
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'))

        conv_1x1_in = ConvModule(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            act_cfg=None
        )
        conv_1x1_out = ConvModule(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm_cfg=dict(type='BN'),
            act_cfg=dict(type='SiLU'))

        conv_3x3_out = None
        if not no_fusion:
            conv_3x3_out = ConvModule(
                in_channels=transformer_dim + in_channels,
                out_channels=in_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='SiLU'))

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0
        num_heads = transformer_dim // head_dim

        ffn_dims = [ffn_dim] * n_transformer_blocks

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dims[block_idx],
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for block_idx in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.dilation = dilation
        self.ffn_max_dim = ffn_dims[0]
        self.ffn_min_dim = ffn_dims[-1]
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w] 需要作图去理解这个过程
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        x = x.contiguous().view(
            info_dict["batch_size"], self.patch_area, info_dict["total_patches"], -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm_conv = self.local_rep(x)

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm_conv) # patches [BP, N , C]

        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        if self.fusion is not None:
            #  for Mvitv3,input + global--> local + global
            fm = self.fusion(torch.cat((fm_conv, fm), dim=1))
        fm = fm + res
        return fm

@BACKBONES.register_module()
class MobileViT_V3(BaseModule):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """
    # Parameters to build layers. 4 parameters are needed to construct a
    # mv2_arch_settings layer, from left to right: expand_ratio, channel, num_blocks, stride.
    # mobilevit_arch_settings layer, from left to right: expand_ratio, out_channels, transformer_channels, ffn_dim, transformer_blocks, patch_h, patch_w,stride, num_heads.
    mv2_arch_settings = [[4, 32, 1, 1], [4, 64, 3, 2]]
    mobilevit_arch_settings=[[4, 128, 144, 288, 2, 2, 2, 2, 4],
                             [4, 256, 192, 384, 4, 2, 2, 2, 4],
                             [4, 320, 240, 480, 3, 2, 2, 2, 4]]

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
                 cls_dropout=0.1
                 ):
        super(MobileViT_V3, self).__init__(init_cfg)
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

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.in_channels=out_channels

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

        for i, mobilevit_layer_cfg in enumerate(self.mobilevit_arch_settings):
            expand_ratio, out_channels, transformer_channels, ffn_dim, \
            transformer_blocks,patch_h, patch_w, stride, num_heads= mobilevit_layer_cfg

            mit_layer, out_channels = self.make_mit_layer(
                input_channel=self.in_channels,
                out_channels=out_channels,
                transformer_channels=transformer_channels,
                ffn_dim=ffn_dim,
                transformer_blocks=transformer_blocks,
                patch_h=patch_h,
                patch_w=patch_w,
                stride=stride,
                num_heads=num_heads,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 3}'
            self.add_module(layer_name, mit_layer)
            self.layers.append(layer_name)
            self.in_channels = out_channels
        self.init_weights()


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


    def make_mit_layer(self, input_channel, out_channels, transformer_channels,ffn_dim, transformer_blocks,patch_h,
                       patch_w, stride, num_heads, expand_ratio)-> [nn.Sequential, int]:
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

        transformer_dim = transformer_channels
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTv3Block(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=transformer_blocks,
            patch_h=patch_h,
            patch_w=patch_w,
            dropout=self.dropout,
            ffn_dropout=self.ffn_dropout,
            attn_dropout=self.attn_dropout,
            head_dim=head_dim,
            no_fusion=False,
            conv_ksize=3
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



    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(MobileViT_V3, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
