import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from .unet_parts import Up, OutConv
from torchvision.models.resnet import ResNet50_Weights
from .se import SEBlock  

class R_SEUNet(nn.Module):  # 修改类名
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(R_SEUNet, self).__init__()
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

        # 替换编码器部分的CBAM为SE
        self.se1 = SEBlock(256)
        self.se2 = SEBlock(512)
        self.se3 = SEBlock(1024)
        self.se4 = SEBlock(2048)
        
        # 替换解码器部分的CBAM为SE
        self.up_se1 = SEBlock(1024)
        self.up_se2 = SEBlock(512)
        self.up_se3 = SEBlock(256)
        self.up_se4 = SEBlock(64)

    def forward(self, x):
        # 编码器部分
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x1 = self.se1(x1)  # 使用SE模块
        
        x2 = self.layer2(x1)
        x2 = self.se2(x2)  # 使用SE模块
        
        x3 = self.layer3(x2)
        x3 = self.se3(x3)  # 使用SE模块
        
        x4 = self.layer4(x3)
        x4 = self.se4(x4)  # 使用SE模块
        
        # 解码器部分
        x = self.up1(x4, x3)
        x = self.up_se1(x)  # 使用SE模块
        
        x = self.up2(x, x2)
        x = self.up_se2(x)  # 使用SE模块
        
        x = self.up3(x, x1)
        x = self.up_se3(x)  # 使用SE模块
        
        x = self.up4(x, x0)
        x = self.up_se4(x)  # 使用SE模块
        
        # 最终上采样到原始尺寸
        x = self.final_upsample(x)
        logits = self.outc(x)
        return logits

if __name__ == '__main__':
    net = R_CBAMUNet(n_channels=1, n_classes=1)
    print(net)