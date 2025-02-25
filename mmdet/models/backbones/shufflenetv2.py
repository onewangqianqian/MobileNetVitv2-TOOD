import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torch.nn.modules.batchnorm import _BatchNorm

# from ..module.activation import act_layers

from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from ..builder import BACKBONES

model_urls = {
    "shufflenetv2_0.5x": "https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth",  # noqa: E501
    "shufflenetv2_1.0x": "https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth",  # noqa: E501
    "shufflenetv2_1.5x": None,
    "shufflenetv2_2.0x": None,
}


def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups, channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class ShuffleV2Block(nn.Module):
    def __init__(self, inp, oup, stride, conv_cfg, norm_cfg, act_cfg=dict(type='ReLU6'),):
        super(ShuffleV2Block, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError("illegal stride value")
        self.stride = stride
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(
                    inp, inp, kernel_size=3, stride=self.stride, padding=1
                ),
                nn.BatchNorm2d(inp),

                ConvModule(
                    in_channels=inp,
                    out_channels=branch_features,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg)
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            ConvModule(
                in_channels=inp if (self.stride > 1) else branch_features,
                out_channels=branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg),

            self.depthwise_conv(
                branch_features,
                branch_features,
                kernel_size=3,
                stride=self.stride,
                padding=1,
            ),
            nn.BatchNorm2d(branch_features),

            ConvModule(
                in_channels=branch_features,
                out_channels=branch_features,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
            )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

@BACKBONES.register_module()
class ShuffleNetV2(BaseModule):
    def __init__(
        self,
        model_size="1.5x",
        out_indices=(2, 3, 4),
        with_last_conv=False,
        kernal_size=3,
        frozen_stages=-1,
        conv_cfg=None,
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='ReLU'),
        norm_eval=False,
        pretrain=False,
    ):
        super(ShuffleNetV2, self).__init__()
        # out_indices can only be a subset of (2, 3, 4)
        assert set(out_indices).issubset((2, 3, 4))

        print("model size is ", model_size)

        self.stage_repeats = [4, 8, 4]
        self.model_size = model_size
        if frozen_stages not in range(-1, 8):
            raise ValueError('frozen_stages must be in range(-1, 8). '
                             f'But received {frozen_stages}')
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.with_last_conv = with_last_conv
        self.kernal_size = kernal_size

        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.pretrain=pretrain
        self.norm_eval = norm_eval

        if model_size == "0.5x":
            self._stage_out_channels = [24, 48, 96, 192, 1024]
        elif model_size == "1.0x":
            self._stage_out_channels = [24, 116, 232, 464, 1024]
        elif model_size == "1.5x":
            self._stage_out_channels = [24, 176, 352, 704, 1024]
        elif model_size == "2.0x":
            self._stage_out_channels = [24, 244, 488, 976, 2048]
        else:
            raise NotImplementedError

        # building first layer
        output_channels = self._stage_out_channels[0]

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_names = ["stage{}".format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
            stage_names, self.stage_repeats, self._stage_out_channels[1:]
        ):
            seq = [
                ShuffleV2Block(
                    input_channels, output_channels, 2, conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
            ]
            for i in range(repeats - 1):
                seq.append(
                    ShuffleV2Block(
                        output_channels, output_channels, 1, conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,act_cfg=self.act_cfg)
                )
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        output_channels = self._stage_out_channels[-1]
        if self.with_last_conv:
            conv5 = ConvModule(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

            self.stage4.add_module("conv5", conv5)
        self.init_weights(self.pretrain)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
            for param in self.maxpool.parameters():
                param.requires_grad = False
            for i in range(2, 5):
                m = getattr(self, "stage{}".format(i))
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        output = []
        for i in range(2, 5):
            stage = getattr(self, "stage{}".format(i))
            x = stage(x)
            if i in self.out_indices:
                output.append(x)
        return tuple(output)

    def init_weights(self, pretrain=False):
        print("init weights...")
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if "first" in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
        if pretrain:
            url = model_urls["shufflenetv2_{}".format(self.model_size)]
            if url is not None:
                pretrained_state_dict = model_zoo.load_url(url)
                print("=> loading pretrained model {}".format(url))
                self.load_state_dict(pretrained_state_dict, strict=False)

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        frozen."""
        super(ShuffleNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()