'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-13 22:01:38
LastEditTime: 2024-11-14 10:50:11
Description: file function description
'''

from models.Backbones.resnet import resnet18, resnet34, resnet50
from models.Backbones.mamba import MambaGraphScan
# from models.Decoders.Decoder import  Seg_Decoder_ResNet, CD_Decoder_ResNet
# from models.Decoders.Decoder2 import SegmentationDecoder, ChangeDetectionDecoder
# from models.Decoders.SegDecoder import SegDecoderResNet
# from models.Decoders.CDDecoder import CD_Decoder_ResNet_Improved
from models.Modules.CIGM import CIGM
from models.Modules.CIEM import CIEM
# from models.Modules.Fre import CFSCDB
# from models.Modules.CA import BCDGuidedSCD_NoDiff
# from models.Modules.IIA import TripleBranchIIA
# from models.Modules.DASPP import BalancedTemporalMSGDC
import torch
from torch import nn
import torch.nn.functional as F

def get_backbone(backbone, pretrained):
    if backbone == 'resnet18':
        backbone = resnet18(pretrained)
    elif backbone == 'resnet34':
        backbone = resnet34(pretrained)
    elif backbone == 'resnet50':
        backbone = resnet50(pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)
    return backbone

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv3x3_dw(in_channel,out_channel,stride=1):
    return nn.Sequential(
        nn.Conv2d(in_channel,in_channel,kernel_size=3,stride=stride,padding=1,groups=in_channel,bias=True),
        nn.BatchNorm2d(in_channel),
        nn.ReLU(),
        nn.Conv2d(in_channel,out_channel,kernel_size=1,stride=1,padding=0,bias=False),
        nn.BatchNorm2d(out_channel),
        nn.ReLU()
    )

class TwoLayerConv2d(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__(nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1, bias=False),
                         nn.BatchNorm2d(in_channels),
                         nn.ReLU(),
                         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   padding=kernel_size // 2, stride=1)
                         )

class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3_dw(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_dw(planes, planes)
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


class EGMSNet(nn.Module):
    def __init__(self, backbone, pretrained, nclass, lightweight, M, Lambda):
        super(EGMSNet, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.lightweight = lightweight
        self.M = M
        self.Lambda = Lambda
        self.backbone = get_backbone(backbone, pretrained)

        if backbone == "resnet18" or backbone == "resnet34":
            self.channel_nums = [64, 128, 256, 512]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        channel_nums = [64, 128, 256, 512]
        # if backbone == "resnet18" or backbone == "resnet34" or backbone == "resnet50":
        self.Seg_Decoder = Seg_Decoder_ResNet(channel_nums)
        self.CD_Decoder = CD_Decoder_ResNet(channel_nums)

        # self.Seg_Decoder = SegmentationDecoder()
        # self.CD_Decoder = ChangeDetectionDecoder()
        # else:
        #     self.Seg_Decoder = Seg_Decoder(self.channel_nums)
        #     self.CD_Decoder = CD_Decoder(self.channel_nums)

        self.CFEM_0 = CFSCDB(64, dw_expand=2, dilations=(1,2), se_ratio=16, freq_expand=2, drop_path=0.1, use_gn=True, freq_ds=2, use_checkpoint=True, fft_dtype=torch.float32)
        self.CFEM_1 = CFSCDB(128, dw_expand=2, dilations=(1,2), se_ratio=16, freq_expand=2, drop_path=0.1, use_gn=True, freq_ds=2, use_checkpoint=True, fft_dtype=torch.float32)
        self.CFEM_2 = CFSCDB(256, dw_expand=2, dilations=(1,2), se_ratio=16, freq_expand=2, drop_path=0.1, use_gn=True, freq_ds=2, use_checkpoint=True, fft_dtype=torch.float32)
        self.CFEM_3 = CFSCDB(512, dw_expand=2, dilations=(1,2), se_ratio=16, freq_expand=2, drop_path=0.1, use_gn=True, freq_ds=2, use_checkpoint=True, fft_dtype=torch.float32)
        # self.CFEM_4 = CIEM(self.channel_nums[4], self.channel_nums[4], self.Lambda)
        self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
        self.DIGM = BCDGuidedSCD_NoDiff(c1=256, c2=256, cg=256,  r_guide=4, norm="gn", share_spatial_mask=True, share_channel_mask=True, spatial_ds=4, spatial_bottleneck_h=0,
        channel_hidden=256 , nonneg_gammas=True)
    #     self.DIGM = BCDGuidedGroupSelectiveMod(channels=256, groups=8, reduction=16,spatial_kernel=7,init_scale=0.5)

        # self.CFEM_0 = CIEM(self.channel_nums[0], self.channel_nums[0], self.Lambda)
        # self.CFEM_1 = CIEM(self.channel_nums[1], self.channel_nums[1], self.Lambda)
        # self.CFEM_2 = CIEM(self.channel_nums[2], self.channel_nums[2], self.Lambda)
        # self.CFEM_3 = CIEM(self.channel_nums[3], self.channel_nums[3], self.Lambda)
        # self.CFEM = [self.CFEM_0, self.CFEM_1, self.CFEM_2, self.CFEM_3]
        # self.DIGM = CIGM(256)


        self.seg_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Dropout(0.2, False),
            nn.Conv2d(128, self.nclass, 1, bias=True)
        )

        self.resCD = self._make_layer(ResBlock, 128, 256, self.M, stride=1)
        self.classifierCD = nn.Sequential(nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
                                          nn.Conv2d(128, 1, kernel_size=1))

        self.mamba_layers2 = MambaGraphScan(dim=256, iters=1, drop_path=0.1)
        self.mamba_layers3 = MambaGraphScan(dim=512, iters=1, drop_path=0.1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes) )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        # resnet34     [0]:(1,64,128,128) [1]:(1,128,64,64) [2]:(1,256,64,64) [3]:(1,512,64,64)
        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)

        feature1_0, feature1_1, feature1_2, feature1_3 = feature1[0], feature1[1], feature1[2], feature1[3]
        feature2_0, feature2_1, feature2_2, feature2_3 = feature2[0], feature2[1], feature2[2], feature2[3]

        feature1_2 = self.mamba_layers2(feature1_2)
        feature2_2 = self.mamba_layers2(feature2_2)

        feature1_3 = self.mamba_layers3(feature1_3)
        feature2_3 = self.mamba_layers3(feature2_3)

        feature1 = feature1_0, feature1_1, feature1_2, feature1_3
        feature2 = feature2_0, feature2_1, feature2_2, feature2_3

        feature_diff = []
        for i in range(len(feature1)):
            feature_diff.append(self.CFEM[i](feature1[i], feature2[i]))     #输入与输出的维度相同




        #CIGM
        out_change = self.resCD(out_change)     #四个输入，一个输出 (1,256,128,128)
        out_1, out_2, out_change = self.DIGM(out_1, out_2, out_change)      #(1,256,128,128)

        # 这后面就是提维度了重点看前面的
        change = F.interpolate(self.classifierCD(out_change), size=(h, w), mode='bilinear', align_corners=False)
        seg1 =  F.interpolate(self.seg_conv(out_1), size=(h, w), mode='bilinear', align_corners=False)
        seg2 = F.interpolate(self.seg_conv(out_2), size=(h, w), mode='bilinear', align_corners=False)

        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EGMSNet(backbone='resnet34', pretrained=True, nclass=7, lightweight=True, M=1, Lambda=0.00005).to(device)
    print(model)
    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)
    from thop import profile
    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))
    with torch.no_grad():
        tc1, tc2, cd = model.forward(image1, image2)
    print(cd.shape, tc1.shape, tc2.shape)