import torch
import torch.nn as nn

__all__ = ["ResBlock"]


class ResBlock(nn.Module):

    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        # self.bn1 = nn.BatchNorm2d(in_channel)
        self.ln1 = nn.LayerNorm(in_channel)
        self.act1 = nn.ReLU(inplace=True)
        # self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False)
        self.conv1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0, bias=False)

        # self.bn2 = nn.BatchNorm2d(in_channel)
        self.ln2 = nn.LayerNorm(in_channel)
        self.act2 = nn.ReLU(inplace=True)
        # self.conv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)

        if in_channel != out_channel:
            # self.bn_shortcut = nn.BatchNorm2d(in_channel)
            self.ln_shortcut = nn.LayerNorm(in_channel)
            self.conv_shortcut = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)
        else:
            # self.bn_shortcut = None
            self.conv_shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # (b, c, h, w)
        # x = self.bn1(x)
        x = self.ln1(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = self.act1(x)
        x = self.conv1(x)

        # x = self.bn2(x)
        x = self.ln2(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        x = self.act2(x)
        x = self.conv2(x)

        if self.conv_shortcut is not None:
            # identity = self.bn_shortcut(identity)
            identity = self.ln_shortcut(identity.permute(0, 2, 3, 1))
            identity = identity.permute(0, 3, 1, 2)
            identity = self.conv_shortcut(identity)
        x = x + identity
        return x
