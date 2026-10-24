import torch
import torch.nn as nn
import torch.nn.functional as F

class conv2d_Resblock(nn.Module):
    def __init__(self, in_channels):
        super(conv2d_Resblock, self).__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        out_conv2d_1 = self.conv2d_1(x)
        out_conv2d_2 = self.conv2d_2(out_conv2d_1)
        out = x + out_conv2d_2
        return out


class Discriminator(nn.Module):
    def __init__(self, inchannel, num_classes, patch_size):
        super(Discriminator, self).__init__()
        dim = 512
        self.patch_size = patch_size
        self.inchannel = inchannel

        self.conv2d_10 = nn.Sequential(
            nn.Conv2d(inchannel, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_base_1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block1 = conv2d_Resblock(64)

        self.conv2d_20 = nn.Sequential(
            nn.Conv2d(inchannel + 64, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_base_2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block2 = conv2d_Resblock(64)

        self.conv2d_30 = nn.Sequential(
            nn.Conv2d(inchannel + 64, 128, kernel_size=(1, 1), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_base_3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3, 3), stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv2d_block3 = conv2d_Resblock(64)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool3 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(192, dim)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(dim, dim)
        self.relu4 = nn.ReLU(inplace=True)
        self.cls_head_src = nn.Linear(dim, num_classes)
        self.netC = nn.Linear(dim, num_classes)

        self.softmax = nn.Softmax(dim=1)


    def _extract_features(self, x):
        out_2dscale1 = self.conv2d_10(x)
        out_2dscale1 = self.conv2d_base_1(out_2dscale1)
        out_2dscale1 = self.conv2d_block1(out_2dscale1)

        out_2dscale2 = torch.cat((out_2dscale1, x), 1)
        out_2dscale2 = self.conv2d_20(out_2dscale2)
        out_2dscale2 = self.conv2d_base_2(out_2dscale2)
        out_2dscale2 = self.conv2d_block2(out_2dscale2)

        out_2dscale3 = torch.cat((out_2dscale2, x), 1)
        out_2dscale3 = self.conv2d_30(out_2dscale3)
        out_2dscale3 = self.conv2d_base_3(out_2dscale3)
        out_2dscale3 = self.conv2d_block3(out_2dscale3)

        out1 = self.avgpool1(out_2dscale1).flatten(1)  # [N, 64]
        out2 = self.avgpool2(out_2dscale2).flatten(1)  # [N, 64]
        out3 = self.avgpool3(out_2dscale3).flatten(1)  # [N, 64]
        return torch.cat((out1, out2, out3), dim=1)  # [N, 192]

    def forward(self, x1, x2, mode='test'):
        features1 = self._extract_features(x1)  # [N, 192]
        features2 = self._extract_features(x2)  # [N, 192]

        out4 = self.relu3(self.fc1(features1))
        out4 = self.relu4(self.fc2(out4))

        out8 = self.relu3(self.fc1(features2))
        out8 = self.relu4(self.fc2(out8))


        if mode == 'test':
            clss = self.cls_head_src(out4)
            return clss
        elif mode == 'train':
            x1_clss = self.cls_head_src(out4)
            x2_clss = self.cls_head_src(out8)
            nume = out8 - out4
            residual_logits = self.netC(nume)
            residual_p = F.softmax(residual_logits, dim=1)
            return x1_clss, x2_clss, residual_p


