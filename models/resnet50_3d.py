from __future__ import annotations

import torch
import torch.nn as nn

from methods import MixStyle3D


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, planes: int, stride: int = 1):
        super().__init__()
        width = planes
        self.conv1 = nn.Conv3d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        return self.relu(out + identity)


class ResNet3D(nn.Module):
    def __init__(
        self,
        block,
        layers: tuple[int, int, int, int],
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 64,
        use_mixstyle: bool = False,
        mixstyle_p: float = 0.2,
        mixstyle_alpha: float = 0.5,
        mixstyle_mix: str = "random",
    ):
        super().__init__()
        self.use_mixstyle = use_mixstyle
        self.inplanes = base_channels
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.mixstyle1 = self._make_mixstyle(use_mixstyle, mixstyle_p, mixstyle_alpha, mixstyle_mix)
        self.mixstyle2 = self._make_mixstyle(use_mixstyle, mixstyle_p, mixstyle_alpha, mixstyle_mix)
        self.mixstyle3 = self._make_mixstyle(use_mixstyle, mixstyle_p, mixstyle_alpha, mixstyle_mix)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.feature_dim = base_channels * 8 * block.expansion
        self.fc = nn.Linear(self.feature_dim, num_classes)
        self._init_weights()

    @staticmethod
    def _make_mixstyle(enabled: bool, p: float, alpha: float, mix: str):
        return MixStyle3D(p=p, alpha=alpha, mix=mix) if enabled else nn.Identity()

    def _make_layer(self, block, planes: int, blocks: int, stride: int):
        layers = [block(self.inplanes, planes, stride=stride)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def set_mixstyle_active(self, active: bool) -> None:
        for module in (self.mixstyle1, self.mixstyle2, self.mixstyle3):
            if isinstance(module, MixStyle3D):
                module.set_activation_status(active)

    def forward_features(self, x):
        x = self.stem(x)
        x = self.mixstyle1(self.layer1(x))
        x = self.mixstyle2(self.layer2(x))
        x = self.mixstyle3(self.layer3(x))
        x = self.layer4(x)
        return self.pool(x).flatten(1)

    def forward(self, x, return_features: bool = False):
        features = self.forward_features(x)
        logits = self.fc(features).squeeze(-1)
        if return_features:
            return logits, features
        return logits


def resnet50_3d(
    num_classes: int = 1,
    in_channels: int = 1,
    use_mixstyle: bool = False,
    mixstyle_p: float = 0.2,
    mixstyle_alpha: float = 0.5,
    mixstyle_mix: str = "random",
    base_channels: int = 64,
    **_kwargs,
):
    return ResNet3D(
        block=Bottleneck3D,
        layers=(3, 4, 6, 3),
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        use_mixstyle=use_mixstyle,
        mixstyle_p=mixstyle_p,
        mixstyle_alpha=mixstyle_alpha,
        mixstyle_mix=mixstyle_mix,
    )


def resnet50(num_classes: int = 1, in_channels: int = 1, **kwargs):
    return resnet50_3d(num_classes=num_classes, in_channels=in_channels, **kwargs)
