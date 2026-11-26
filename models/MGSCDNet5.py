"""Ding, Lei, Kun Zhu, Daifeng Peng, Hao Tang, Kuiwu Yang, and Lorenzo Bruzzone. 2024. “Joint Spatio-Temporal Modeling for Semantic
Change Detection in Remote Sensing Images.” IEEE Transactions on Geoscience and
Remote Sensing 62:1–11. https://doi.org/10.1109/TGRS.2024.3368168."""
import torch
import torch.nn as nn
from torchvision import models
from torch.nn import functional as F
# from utils.misc import initialize_weights
from Compare.bone.CSWin_Transformer import mit
from models.Backbones.mamba import MambaGraphScan
# from models.Modules.Fre import EnhancedFreqBlock
from models.Modules.Fre2 import BiTemporalXMix
from models.Decoders.Decoder import Seg_Decoder_ResNet, CD_Decoder_ResNet
from models.Modules.CA import BCDGuidedSCD_NoDiff
from  models.Modules.CASAB import CBAM

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

args = {'hidden_size': 128 * 3,
        'mlp_dim': 256 * 3,
        'num_heads': 4,
        'num_layers': 2,
        'dropout_rate': 0.}


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low // scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low // scale_ratio),
            nn.BatchNorm2d(in_channels_low // scale_ratio),
            nn.ReLU(inplace=True))
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat((x, low_feat), dim=1)
        x = self.decode(x)
        return x


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained=pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        self.head = nn.Sequential(nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.BatchNorm2d(128), nn.ReLU())
        initialize_weights(self.head)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MGSCDNet5(nn.Module):
    def __init__(self, in_channels=3, num_classes=7, input_size=512):
        super(MGSCDNet5, self).__init__()
        feat_size = input_size // 4
        self.FCN = FCN(in_channels, pretrained=True)

        self.classifierA = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierB = nn.Conv2d(128, num_classes, kernel_size=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(128, 64, kernel_size=1), nn.BatchNorm2d(64), nn.ReLU(),
                                          nn.Conv2d(64, 1, kernel_size=1))

        initialize_weights( self.classifierA, self.classifierB, self.classifierCD)

        self.mamba_layers1 = MambaGraphScan(dim=64, iters=1, drop_path=0.1)
        self.mamba_layers2 = MambaGraphScan(dim=128, iters=1, drop_path=0.1)
        self.mamba_layers3 = MambaGraphScan(dim=256, iters=1, drop_path=0.1)
        self.mamba_layers4 = MambaGraphScan(dim=512, iters=1, drop_path=0.1)
    #
        self.SR1 = CBAM(64, reduction=16, spatial_kernel=7)
        self.SR2 = CBAM(128, reduction=16, spatial_kernel=7)
        self.SR3 = CBAM(256, reduction=16, spatial_kernel=7)
        self.SR4 = CBAM(512, reduction=16, spatial_kernel=7)
    #
        self.CFEM_0 = BiTemporalXMix(c=64,dw_expand=2,dilations=(1, 3, 6), ca_ratio=8,freq_expand=2)
        self.CFEM_1 = BiTemporalXMix(c=128,dw_expand=2,dilations=(1, 3, 6), ca_ratio=8,freq_expand=2)
        self.CFEM_2 = BiTemporalXMix(c=256,dw_expand=2,dilations=(1, 3, 6), ca_ratio=8,freq_expand=2)
        self.CFEM_3 = BiTemporalXMix(c=512,dw_expand=2,dilations=(1, 3, 6), ca_ratio=8,freq_expand=2)
    #
    #     # if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50":
        self.Seg_Decoder = Seg_Decoder_ResNet(channel_nums=(64,128,256,512))
        self.CD_Decoder = CD_Decoder_ResNet(channel_nums=(64,128,256,512))
    #
    #     self.DIGM = BCDGuidedSCD_NoDiff(
    #     channels1=128,
    #     channels2=128,
    #     guide_channels=128,
    #     guide_reduction=4,
    #     norm="gn",
    #     share_spatial_mask=True,
    #     share_channel_mask=True,
    #     spatial_downsample=2,
    #     spatial_bottleneck_dim=0,
    #     channel_hidden_dim=128,
    #     non_negative_gammas=True,
    # )

    def base_forward(self, x):

        x = self.FCN.layer0(x)  # size:1/2             (1,64,256,256)
        x = self.FCN.maxpool(x)  # size:1/4            (1,64,128,128)
        x1 = self.FCN.layer1(x)  # size:1/4         (1,64,128,128)
        x1 = self.mamba_layers1(x1)

        x2 = self.FCN.layer2(x1)  # size:1/8         (1,128,64,64)
        x2 = self.mamba_layers2(x2)

        x3 = self.FCN.layer3(x2)                         #(1,256,64,64)
        x3 = self.FCN.maxpool(x3)
        x3 = self.mamba_layers3(x3)

        x4 = self.FCN.layer4(x3)                         #(1,512,64,64)
        x4 = self.mamba_layers4(x4)

        return x1, x2, x3, x4

    def forward(self, x1, x2):
        x_size = x1.size()

        feature1_0, feature1_1, feature1_2, feature1_3 = self.base_forward(x1)

        feature2_0, feature2_1, feature2_2, feature2_3 = self.base_forward(x2)

        feature_diff1 = self.CFEM_0(feature1_0, feature2_0)
        feature_diff2 = self.CFEM_1(feature1_1, feature2_1)
        feature_diff3 = self.CFEM_2(feature1_2, feature2_2)
        feature_diff4 = self.CFEM_3(feature1_3, feature2_3)
        feature_diff = [feature_diff1, feature_diff2, feature_diff3, feature_diff4]


        feature1_0 = self.SR1(feature1_0)
        feature1_1 = self.SR2(feature1_1)
        feature1_2 = self.SR3(feature1_2)
        feature1_3 = self.SR4(feature1_3)
        feature1 = [feature1_0, feature1_1, feature1_2, feature1_3]

        feature2_0 = self.SR1(feature2_0)
        feature2_1 = self.SR2(feature2_1)
        feature2_2 = self.SR3(feature2_2)
        feature2_3 = self.SR4(feature2_3)
        feature2 = [feature2_0, feature2_1, feature2_2, feature2_3]

        #SegDecoder
        out_1 = self.Seg_Decoder(feature1)     #(1,256,128,128)
        out_2 = self.Seg_Decoder(feature2)     #(1,256,128,128)

        #CDDecoder
        out_change = self.CD_Decoder(feature_diff)     #(1,256,128,128)

        out1 = self.classifierA(out_1)
        out2 = self.classifierB(out_2)
        change = self.classifierCD(out_change)

        out1 = F.upsample(out1, x_size[2:], mode='bilinear')
        out2 = F.upsample(out2, x_size[2:], mode='bilinear')
        change = F.upsample(change, x_size[2:], mode='bilinear')
        change = torch.sigmoid(change)

        return out1, out2, change.squeeze(1)




if __name__ == '__main__':
    test_data1 = torch.rand(1, 3, 512, 512).cuda()
    test_data2 = torch.rand(1, 3, 512, 512).cuda()
    model = MGSCDNet5(3, num_classes=7).cuda()
    model = model.cuda()
    changes, preds1, preds2 = model(test_data1, test_data2)
    print(changes.shape)
    print(preds1.shape)
    print(preds2.shape)
