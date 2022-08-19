import torch
import torch.nn as nn

__all__ = ["ResBlock"]


class ResBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.bn1 = nn.BatchNorm2d(in_channel)
        self.act1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False) # TODO check kernel size 3 or 1
        self.conv1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False)

        self.bn2 = nn.BatchNorm2d(in_channel)
        self.act2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)

        if in_channel != out_channel:
            self.bn_shortcut = nn.BatchNorm2d(in_channel)
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)
        else:
            self.bn_shortcut = None
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # (b, c, h, w)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.act2(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            identity = self.bn_shortcut(identity)
            identity = self.conv_shortcut(identity)
        x = x + identity
        return x
