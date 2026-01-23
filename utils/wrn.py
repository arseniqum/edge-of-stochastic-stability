import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class WideResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = _conv3x3(out_channels, out_channels, stride=1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        out += self.shortcut(x)
        return out


def _make_group(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
) -> nn.Sequential:
    blocks = [WideResNetBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(WideResNetBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*blocks)


class WideResNet(nn.Module):
    """
    Wide ResNet implementation for CIFAR-style inputs.

    depth must satisfy depth = 6 * n + 4 for some integer n >= 1.
    widen_factor controls the channel multiplier relative to the base network.
    """

    def __init__(self, depth: int = 10, widen_factor: int = 2, num_classes: int = 10):
        super().__init__()

        if (depth - 4) % 6 != 0 or (depth - 4) // 6 <= 0:
            raise ValueError(
                "depth must be of the form 6n+4 with n>=1 (e.g., 10, 16, 22, ...). "
                f"Got depth={depth}."
            )
        if widen_factor < 1:
            raise ValueError(f"widen_factor must be >= 1, got {widen_factor}.")

        num_blocks_per_group = (depth - 4) // 6
        widths = [16, 32, 64]
        widths = [channel * widen_factor for channel in widths]

        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.group1 = _make_group(16, widths[0], num_blocks_per_group, stride=1)
        self.group2 = _make_group(widths[0], widths[1], num_blocks_per_group, stride=2)
        self.group3 = _make_group(widths[1], widths[2], num_blocks_per_group, stride=2)

        self.bn = nn.BatchNorm2d(widths[2])
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)

    def forward(self, x):
        out = self.stem(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.relu(self.bn(out))
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class FixupWideResNetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()

        self.bias1a = nn.Parameter(torch.zeros(1, in_channels, 1, 1))
        self.bias1b = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.bias2a = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.bias2b = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, out_channels, 1, 1))

        self.conv1 = _conv3x3(in_channels, out_channels, stride=stride)
        self.conv2 = _conv3x3(out_channels, out_channels, stride=1)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=stride, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(F.relu(x + self.bias1a, inplace=True))
        out = out + self.bias1b
        out = self.conv2(F.relu(out + self.bias2a, inplace=True))
        out = out * self.scale + self.bias2b
        out = out + self.shortcut(x)
        return out


def _make_fixup_group(
    in_channels: int,
    out_channels: int,
    num_blocks: int,
    stride: int,
) -> nn.Sequential:
    blocks = [FixupWideResNetBlock(in_channels, out_channels, stride=stride)]
    for _ in range(1, num_blocks):
        blocks.append(FixupWideResNetBlock(out_channels, out_channels, stride=1))
    return nn.Sequential(*blocks)


class WideResNetNoBN(nn.Module):
    """
    Wide ResNet variant without BatchNorm using Fixup initialization.
    """

    def __init__(self, depth: int = 10, widen_factor: int = 2, num_classes: int = 10):
        super().__init__()

        if (depth - 4) % 6 != 0 or (depth - 4) // 6 <= 0:
            raise ValueError(
                "depth must be of the form 6n+4 with n>=1 (e.g., 10, 16, 22, ...). "
                f"Got depth={depth}."
            )
        if widen_factor < 1:
            raise ValueError(f"widen_factor must be >= 1, got {widen_factor}.")

        num_blocks_per_group = (depth - 4) // 6
        widths = [16, 32, 64]
        widths = [channel * widen_factor for channel in widths]

        self.stem = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)

        self.group1 = _make_fixup_group(16, widths[0], num_blocks_per_group, stride=1)
        self.group2 = _make_fixup_group(widths[0], widths[1], num_blocks_per_group, stride=2)
        self.group3 = _make_fixup_group(widths[1], widths[2], num_blocks_per_group, stride=2)

        self.bias = nn.Parameter(torch.zeros(1, widths[2], 1, 1))
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[2], num_classes)

        self.num_blocks_per_group = num_blocks_per_group
        self.num_residual_blocks = num_blocks_per_group * 3

    def forward(self, x):
        out = self.stem(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.relu(out + self.bias)
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def initialize_wrn_fixup(net: WideResNetNoBN):
    """
    Apply Fixup initialization scheme to a WideResNet without BatchNorm.
    """
    num_residual_blocks = net.num_residual_blocks

    nn.init.kaiming_normal_(net.stem.weight, mode="fan_in", nonlinearity="relu")

    for module in net.modules():
        if isinstance(module, FixupWideResNetBlock):
            nn.init.kaiming_normal_(module.conv1.weight, mode="fan_in", nonlinearity="relu")
            module.conv1.weight.data.mul_(num_residual_blocks ** (-0.5))
            nn.init.zeros_(module.conv2.weight)
            if isinstance(module.shortcut, nn.Conv2d):
                nn.init.kaiming_normal_(module.shortcut.weight, mode="fan_in", nonlinearity="relu")

    nn.init.zeros_(net.fc.weight)
    nn.init.zeros_(net.fc.bias)
