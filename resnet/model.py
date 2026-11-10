"""ResNet for CIFAR-10, as described in Section 4.2 of He et al. (2015).

The network has 6n + 2 weighted layers:

    stem     3x3 conv, 16 filters                        32x32
    stage1   n BasicBlocks, 16 filters                   32x32
    stage2   n BasicBlocks, 32 filters, first stride 2   16x16
    stage3   n BasicBlocks, 64 filters, first stride 2    8x8
    head     global average pooling + linear layer

n = 3, 5, 7, 9, 18 gives ResNet-20, 32, 44, 56, 110.
"""

from torch import Tensor, nn


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """3x3 convolution that keeps the spatial size, or halves it with stride 2."""
    # Every conv is followed by BatchNorm, whose learned shift makes a conv bias redundant.
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution, used to project the shortcut when the shape changes."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Two 3x3 convolutions and a shortcut: ``y = relu(F(x) + shortcut(x))``.

    The shortcut is the identity when input and output shapes match, and a strided
    1x1 projection otherwise (option B in the paper). With ``residual=False`` there is
    no shortcut at all, which gives the plain block used as a baseline.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, residual: bool = True):
        super().__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample: nn.Module | None
        if not residual:
            self.downsample = None
        elif stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, out_channels, stride), nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            out = out + self.downsample(x)
        return self.relu(out)


def make_stage(
    in_channels: int, out_channels: int, num_blocks: int, stride: int, residual: bool = True
) -> nn.Sequential:
    """Stack ``num_blocks`` blocks. Only the first one may change channels or resolution."""
    blocks = [BasicBlock(in_channels, out_channels, stride, residual)]
    blocks += [BasicBlock(out_channels, out_channels, 1, residual) for _ in range(num_blocks - 1)]
    return nn.Sequential(*blocks)


class ResNet(nn.Module):
    """CIFAR-10 ResNet with ``depth = 6n + 2`` weighted layers.

    Args:
        depth: number of weighted layers, e.g. 20, 32, 44, 56 or 110.
        num_classes: size of the classification layer.
        residual: build the plain counterpart (same layers, no shortcuts) when False.
        zero_init_residual: set the last BatchNorm scale of every block to zero, so each
            block starts out as an identity mapping (Goyal et al., 2017).
    """

    def __init__(
        self,
        depth: int = 20,
        num_classes: int = 10,
        residual: bool = True,
        zero_init_residual: bool = False,
    ):
        super().__init__()
        if depth < 8 or (depth - 2) % 6 != 0:
            raise ValueError(f"depth must be 6n + 2 with n >= 1 (20, 32, 44, 56, ...), got {depth}")
        n = (depth - 2) // 6
        self.depth = depth
        self.residual = residual

        self.stem = nn.Sequential(conv3x3(3, 16), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.stage1 = make_stage(16, 16, n, stride=1, residual=residual)
        self.stage2 = make_stage(16, 32, n, stride=2, residual=residual)
        self.stage3 = make_stage(32, 64, n, stride=2, residual=residual)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, num_classes))

        self._init_weights(zero_init_residual=zero_init_residual and residual)

    def _init_weights(self, zero_init_residual: bool) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)
