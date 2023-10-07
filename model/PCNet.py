import torch
import torch.nn as nn
import torch.nn.functional as F

from model.res2net import res2net50_v1b_26w_4s


class BasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1
    ):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


class NeighborConnectionDecoder(nn.Module):
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1, x2, x3):
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(x2_1))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x


class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel // subchannel
        self.conv = nn.Sequential(
            nn.Conv2d(channel + 2 * self.group, channel, 3, padding=1),
            nn.BatchNorm2d(channel),
            nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y, z):
        if self.group == 1:
            x_cat = torch.cat((x, y, z), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, z, xs[1], y, z), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, z, xs[1], y, z, xs[2], y, z, xs[3], y, z), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat(
                (
                    xs[0],y,z,xs[1],y,z,xs[2],y,z,xs[3],y,z,xs[4],y,z,xs[5],y,z,xs[6],y,z,xs[7],y,z,
                ),
                1,
            )
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat(
                (
                    xs[0],y,z,xs[1],y,z,xs[2],y,z,xs[3],y,z,xs[4],y,z,xs[5],y,z,xs[6],y,z,xs[7],y,z,xs[8],y,z,xs[9],y,z,xs[10],y,z,xs[11],y,z,xs[12],y,z,xs[13],y,z,xs[14],y,z,xs[15],y,z,
                ),
                1,
            )
        else:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat(
                (
                    xs[0],y,z,xs[1],y,z,xs[2],y,z,xs[3],y,z,xs[4],y,z,xs[5],y,z,xs[6],y,z,xs[7],y,z,xs[8],y,z,xs[9],y,z,xs[10],y,z,xs[11],y,z,xs[12],y,z,xs[13],y,z,xs[14],y,z,xs[15],y,z,xs[16],y,z,xs[17],y,z,xs[18],y,z,xs[19],y,z,xs[20],y,z,xs[21],y,z,xs[22],y,z,xs[23],y,z,xs[24],y,z,xs[25],y,z,xs[26],y,z,xs[27],y,z,xs[28],y,z,xs[29],y,z,xs[30],y,z,xs[31],y,z,
                ),
                1,
            )

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y


class ReverseStage(nn.Module):
    def __init__(self, channel):
        super(ReverseStage, self).__init__()
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)
        self.ga = GA(channel)

    def forward(self, x, y, z):
        y = -1 * (torch.sigmoid(y)) + 1
        x = self.GA(x, z)

        x, y = self.weak_gra(x, y, z)
        x, y = self.medium_gra(x, y, z)
        _, y = self.strong_gra(x, y, z)

        return y


class ConvBNR(nn.Module):
    def __init__(
        self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False
    ):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                inplanes,
                planes,
                kernel_size,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                bias=bias,
            ),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EC(nn.Module):
    def __init__(self, channel):
        super(EC, self).__init__()
        self.block = nn.Sequential(
            ConvBNR(channel * 2, channel * 2, 3),
            ConvBNR(channel * 2, channel, 3),
            Conv1x1(channel, 1),
        )

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x4 = F.interpolate(x4, size, mode="bilinear", align_corners=False)
        temp = x1 * x4
        x1 = x1 + temp
        x4 = x4 + temp
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out


class GA(nn.Module):
    def __init__(self, channel):
        super(GA, self).__init__()
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = Conv1x1(channel, channel)

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode="bilinear", align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        weit = self.avg_pool(x)
        x = x * weit + x
        x = self.conv1d(x)

        return x


class Network(nn.Module):
    def __init__(self, cfg, channel=32, imagenet_pretrained=True):
        super(Network, self).__init__()
        self.cfg = cfg
        self.bkbone = res2net50_v1b_26w_4s(self.cfg, pretrained=imagenet_pretrained)
        self.rfb1_1 = RFB_modified(256, channel)
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)
        self.NCD = NeighborConnectionDecoder(channel)
        self.RS5 = ReverseStage(channel)
        self.RS4 = ReverseStage(channel)
        self.RS3 = ReverseStage(channel)
        self.RS2 = ReverseStage(channel)
        self.ec = EC(channel)

    def forward(self, x):
        x = self.bkbone.conv1(x)
        x = self.bkbone.bn1(x)
        x = self.bkbone.relu(x)
        x = self.bkbone.maxpool(x)
        x1 = self.bkbone.layer1(x)
        x2 = self.bkbone.layer2(x1)
        x3 = self.bkbone.layer3(x2)
        x4 = self.bkbone.layer4(x3)

        x1_rfb = self.rfb1_1(x1)
        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(
            S_g, scale_factor=8, mode="bilinear"
        )

        edge = self.ec(x4_rfb, x1_rfb)
        S_edge = F.interpolate(
            edge, scale_factor=4, mode="bilinear", align_corners=False
        )

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode="bilinear")
        guidance_edge = F.interpolate(edge, scale_factor=0.125, mode="bilinear")
        ra4_feat = self.RS5(x4_rfb, guidance_g, guidance_edge)
        S_5 = ra4_feat + guidance_g
        S_5_pred = F.interpolate(
            S_5, scale_factor=32, mode="bilinear"
        )

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode="bilinear")
        guidance_edge = F.interpolate(edge, scale_factor=0.25, mode="bilinear")
        ra3_feat = self.RS4(x3_rfb, guidance_5, guidance_edge)
        S_4 = ra3_feat + guidance_5
        S_4_pred = F.interpolate(
            S_4, scale_factor=16, mode="bilinear"
        )

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode="bilinear")
        guidance_edge = F.interpolate(edge, scale_factor=0.5, mode="bilinear")
        ra2_feat = self.RS3(x2_rfb, guidance_4, guidance_edge)
        S_3 = ra2_feat + guidance_4
        S_3_pred = F.interpolate(
            S_3, scale_factor=8, mode="bilinear"
        )

        # ---- reverse stage 2 ----
        guidance_3 = F.interpolate(S_3, scale_factor=2, mode="bilinear")
        guidance_edge = edge
        ra1_feat = self.RS2(x1_rfb, guidance_3, guidance_edge)
        S_2 = ra1_feat + guidance_3
        S_2_pred = F.interpolate(
            S_2, scale_factor=4, mode="bilinear"
        )
        return S_g_pred, S_5_pred, S_4_pred, S_3_pred, S_2_pred, S_edge
