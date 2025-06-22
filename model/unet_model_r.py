import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from .unet_parts import Up, OutConv
# 在文件顶部添加导入
from torchvision.models.resnet import ResNet50_Weights

class ResNet50UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(ResNet50UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 使用ResNet50作为编码器
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        
        # 修改第一层卷积以匹配输入通道数
        if n_channels != 3:
            self.resnet.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 获取中间特征图
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.layer1 = nn.Sequential(self.resnet.maxpool, self.resnet.layer1)
        self.layer2 = self.resnet.layer2
        self.layer3 = self.resnet.layer3
        self.layer4 = self.resnet.layer4

        # 解码器部分
        self.up1 = Up(2048, 1024, bilinear)
        self.up2 = Up(1024, 512, bilinear)
        self.up3 = Up(512, 256, bilinear)
        self.up4 = Up(256, 64, bilinear)
        
        # 添加最终上采样层
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # 编码器部分
        x0 = self.layer0(x)  # [b,64,h/2,w/2]
        x1 = self.layer1(x0) # [b,256,h/4,w/4]
        x2 = self.layer2(x1) # [b,512,h/8,w/8]
        x3 = self.layer3(x2) # [b,1024,h/16,w/16]
        x4 = self.layer4(x3) # [b,2048,h/32,w/32]
        
        # 解码器部分
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        
        # 最终上采样到原始尺寸
        x = self.final_upsample(x)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = ResNet50UNet(n_channels=1, n_classes=1)
    print(net)