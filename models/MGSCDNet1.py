'''
Author: Zuoxibing
email: zuoxibing1015@163.com
Date: 2024-11-13 22:01:38
LastEditTime: 2024-11-14 10:50:11
Description: Very Weak Baseline EGMSNet for ablation study
'''

from models.Backbones.resnet import resnet18, resnet34, resnet50
import torch
from torch import nn
import torch.nn.functional as F

def    get_backbone(backbone, pretrained):
    if backbone == 'resnet18':
        backbone = resnet18(pretrained)
    elif backbone == 'resnet34':
        backbone = resnet34(pretrained)
    elif backbone == 'resnet50':
        backbone = resnet50(pretrained)
    else:
        exit("\nError: BACKBONE \'%s\' is not implemented!\n" % backbone)
    return backbone

class Ultra_Simple_Decoder(nn.Module):
    """极度简化的解码器"""
    def __init__(self, channel_nums):
        super().__init__()
        self.channel_nums = channel_nums
        self.channel = 32  # 极小的通道数

        # 只用一个卷积层融合所有特征
        total_channels = sum(channel_nums)  # 64+128+256+512=960
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(total_channels, self.channel, 1, padding=2),  # 使用5x5大卷积核，计算量大但效果可能不好
            nn.Dropout(0.1),  # 极高的dropout
            nn.ReLU()
        )

    def forward(self, layers):
        layer0, layer1, layer2, layer3 = layers[0], layers[1], layers[2], layers[3]

        # 简单粗暴地将所有特征上采样到最大尺寸然后拼接
        target_size = layer0.shape[2:]  # 128x128

        layer1_up = F.interpolate(layer1, size=target_size, mode='nearest')
        layer2_up = F.interpolate(layer2, size=target_size, mode='nearest')
        layer3_up = F.interpolate(layer3, size=target_size, mode='nearest')

        # 直接拼接所有特征
        fused = torch.cat([layer0, layer1_up, layer2_up, layer3_up], dim=1)
        out = self.fusion_conv(fused)

        return out

class MGSCDNet1(nn.Module):
    """极度弱化的baseline模型"""
    def __init__(self, backbone, pretrained, nclass, M=1):  # 只用1个ResBlock
        super(MGSCDNet1, self).__init__()
        self.backbone_name = backbone
        self.nclass = nclass
        self.M = M
        self.backbone = get_backbone(backbone, pretrained)

        if backbone == "resnet18" or backbone == "resnet34":
            self.channel_nums = [64, 128, 256, 512]
        elif backbone == "resnet50":
            self.channel_nums = [256, 512, 1024, 2048]

        channel_nums = [64, 128, 256, 512]

        # 使用极简解码器
        self.Feature1_Decoder = Ultra_Simple_Decoder(channel_nums)
        self.Feature2_Decoder = Ultra_Simple_Decoder(channel_nums)
        self.Diff_Decoder = Ultra_Simple_Decoder(channel_nums)

        # 极简分割头 - 移除dropout
        self.seg_conv = nn.Sequential(
            nn.Conv2d(32, 16, 1),  # 使用1x1卷积
            nn.ReLU(),
            nn.Conv2d(16, self.nclass, 1)
        )

        # 极简变化检测头 - 移除dropout
        self.resCD = nn.Sequential(
            nn.Conv2d(32, 32, 1),  # 使用1x1卷积
            nn.ReLU()
        )

        self.classifierCD = nn.Sequential(
            nn.Conv2d(32, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        feature1 = self.backbone.forward(x1)
        feature2 = self.backbone.forward(x2)

        feature1_0, feature1_1, feature1_2, feature1_3 = feature1[0], feature1[1], feature1[2], feature1[3]
        feature2_0, feature2_1, feature2_2, feature2_3 = feature2[0], feature2[1], feature2[2], feature2[3]

        # 最简单的差异计算
        diff1 = feature1_0 - feature2_0  # 直接相减，不取绝对值
        diff2 = feature1_1 - feature2_1
        diff3 = feature1_2 - feature2_2
        diff4 = feature1_3 - feature2_3

        # 三个极简解码器
        out1 = self.Feature1_Decoder([feature1_0, feature1_1, feature1_2, feature1_3])
        out2 = self.Feature2_Decoder([feature2_0, feature2_1, feature2_2, feature2_3])
        out_change = self.Diff_Decoder([diff1, diff2, diff3, diff4])

        # 极简后处理
        out_change_processed = self.resCD(out_change)

        # 使用最差的上采样方式
        change = F.interpolate(self.classifierCD(out_change_processed), size=(h, w), mode='nearest')
        seg1 = F.interpolate(self.seg_conv(out1), size=(h, w), mode='nearest')
        seg2 = F.interpolate(self.seg_conv(out2), size=(h, w), mode='nearest')

        change = torch.sigmoid(change)

        return seg1, seg2, change.squeeze(1)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGSCDNet1(backbone='resnet34', pretrained=True, nclass=7, M=1).to(device)
    print(model)

    image1 = torch.randn(1, 3, 512, 512).to(device)
    image2 = torch.randn(1, 3, 512, 512).to(device)

    from thop import profile
    FLOPs, Params = profile(model, (image1, image2))
    print('Params = %.2f M, FLOPs = %.2f G' % (Params / 1e6, FLOPs / 1e9))

    with torch.no_grad():
        tc1, tc2, cd = model.forward(image1, image2)
    print(f'Output shapes - CD: {cd.shape}, Seg1: {tc1.shape}, Seg2: {tc2.shape}')