import torch
import torch.nn as nn

__all__ = ["ResBlock", "Encoder"]

#
# class ResBlock(nn.Module):
#
#     def __init__(self, in_channel: int, out_channel: int):
#         super().__init__()
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#
#         self.bn1 = nn.BatchNorm2d(in_channel)
#         self.act1 = nn.ReLU(inplace=True)
#         # self.conv1 = nn.Conv2d(in_channel, in_channel, 3, 1, 1, bias=False) # TODO check kernel size 3 or 1
#         self.conv1 = nn.Conv2d(in_channel, out_channel//4, 1, 1, 0, bias=False) # bottleneck
#
#         self.bn2 = nn.BatchNorm2d(in_channel)
#         self.act2 = nn.ReLU(inplace=True)
#         # self.conv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1, bias=True)
#         self.conv2 = nn.Conv2d(out_channel//4, out_channel, 1, 1, 0, bias=True)
#
#         if in_channel != out_channel:
#             self.bn_shortcut = nn.BatchNorm2d(in_channel)
#             self.conv_shortcut = nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=True)
#         else:
#             self.bn_shortcut = None
#             self.conv_shortcut = None
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         identity = x  # (b, c, h, w)
#         x = self.bn1(x)
#         x = self.act1(x)
#         x = self.conv1(x)
#
#         x = self.bn2(x)
#         x = self.act2(x)
#         x = self.conv2(x)
#
#         if self.conv_shortcut is not None:
#             identity = self.bn_shortcut(identity)
#             identity = self.conv_shortcut(identity)
#         x = x + identity
#         return x


class Encoder(nn.Module):
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        elif stride == 1:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )
        elif stride == 1:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 3, stride=1, padding=1)
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class ResBlock(nn.Module):

    def __init__(self, in_channel: int, channel: int):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = channel

        self.act1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channel, channel, 3, padding=1) # bottleneck

        self.act2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channel, in_channel, 1)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x  # (b, c, h, w)
        x = self.act1(x)
        x = self.conv1(x)

        x = self.act2(x)
        x = self.conv2(x)

        x = x + identity
        return x
