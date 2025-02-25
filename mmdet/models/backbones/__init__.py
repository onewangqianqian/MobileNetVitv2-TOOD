# Copyright (c) OpenMMLab. All rights reserved.
from .csp_darknet import CSPDarknet
from .darknet import Darknet
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
from .efficientnet import EfficientNet
from .hourglass import HourglassNet
from .hrnet import HRNet
from .mobilenet_v2 import MobileNetV2
from .pvt import PyramidVisionTransformer, PyramidVisionTransformerV2
from .regnet import RegNet
from .res2net import Res2Net
from .resnest import ResNeSt
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .swin import SwinTransformer
from .trident_resnet import TridentResNet
# from .shufflenetv2 import ShuffleNetV2
from .mobilevit import MobileViT
from .mobilevit_v3 import  MobileViT_V3
# from .mobilevit_v3a import MobileViT_V3a
# from .mobilevit_v3b import MobileViT_V3b
from .mobilevit_v2 import MobileViTv2
# from .efficientvit import EfficientViT_M4


__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet',
    'MobileNetV2', 'Res2Net', 'HourglassNet', 'DetectoRS_ResNet',
    'DetectoRS_ResNeXt', 'Darknet', 'ResNeSt', 'TridentResNet', 'CSPDarknet',
    'SwinTransformer', 'PyramidVisionTransformer',
    'PyramidVisionTransformerV2', 'EfficientNet','MobileViT','MobileViT_V3',
    'MobileViTv2'
]
# 'ShuffleNetV2'
#'EfficientViT_M4','MobileViT_V3a', 'MobileViT_V3b',