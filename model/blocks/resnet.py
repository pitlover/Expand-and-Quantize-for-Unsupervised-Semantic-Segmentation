import torch
import torch.nn as nn

__all__ = ["EncResBlock", "DecResBlock", "LayerNorm2d", "ResBlock"]


class LayerNorm2d(nn.LayerNorm):

    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor):
        # b, c, h, w
        x = x.permute(0, 2, 3, 1).contiguous()
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class EncResBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        # self.norm1 = nn.BatchNorm2d(in_channel)
        # self.norm1 = nn.GroupNorm(num_groups=16, num_channels=in_channel)
        # self.norm1 = LayerNorm2d(in_channel)
        self.norm1 = nn.Identity()

        self.act1 = nn.ReLU()
        # self.act1 = nn.ReLU(inplace=True)
        # self.act1 = nn.LeakyReLU(0.1, inplace=True)  # TODO check
        # self.act1 = nn.Identity()

        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)

        # self.norm2 = nn.BatchNorm2d(out_channel)
        # self.norm2 = nn.BatchNorm2d(in_channel // 16)
        # self.norm2 = nn.GroupNorm(num_groups=16, num_channels=in_channel)
        # self.norm2 = LayerNorm2d(in_channel)
        self.norm2 = nn.Identity()

        self.act2 = nn.ReLU()
        # self.act2 = nn.ReLU(inplace=True)
        # self.act2 = nn.LeakyReLU(0.1, inplace=True)

        # self.conv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=True)

        if in_channel != out_channel:
            # self.norm_shortcut = nn.BatchNorm2d(in_channel)
            # self.norm_shortcut = nn.GroupNorm(num_groups=16, num_channels=in_channel)
            # self.norm_shortcut = LayerNorm2d(in_channel)
            self.norm_shortcut = nn.Identity()
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        else:
            self.norm_shortcut = None
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # (b, c, h, w)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            identity = self.norm_shortcut(identity)
            identity = self.conv_shortcut(identity)
        x = x + identity

        return x


class DecResBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.norm1 = nn.BatchNorm2d(in_channel)
        # self.norm1 = nn.GroupNorm(num_groups=16, num_channels=in_channel)
        # self.norm1 = LayerNorm2d(in_channel)
        # self.norm1 = nn.Identity()

        # self.act1 = nn.ReLU(inplace=True)
        self.act1 = nn.LeakyReLU(0.1, inplace=True)
        # self.act1 = nn.Identity()

        # self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)

        self.norm2 = nn.BatchNorm2d(out_channel)
        # self.norm2 = nn.GroupNorm(num_groups=16, num_channels=out_channel)
        # self.norm2 = LayerNorm2d(in_channel)
        # self.norm2 = nn.Identity()

        # self.act2 = nn.ReLU(inplace=True)
        self.act2 = nn.LeakyReLU(0.1, inplace=True)

        # self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 1, 1, 0, bias=True)

        if in_channel != out_channel:
            self.norm_shortcut = nn.BatchNorm2d(in_channel)
            # self.norm_shortcut = nn.GroupNorm(num_groups=16, num_channels=in_channel)
            # self.norm_shortcut = LayerNorm2d(in_channel)
            # self.norm_shortcut = nn.Identity()

            # self.act_shortcut = nn.ReLU(inplace=True)  # TODO check
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False)
            # self.conv_shortcut = nn.Identity()
        else:
            self.norm_shortcut = None
            self.conv_shortcut = None
            self.act_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # (b, c, h, w)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            identity = self.norm_shortcut(identity)
            identity = self.conv_shortcut(identity)

        x = x + identity
        return x

class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            # nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input
        return out