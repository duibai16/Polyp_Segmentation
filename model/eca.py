import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

class ECABlock(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECABlock, self).__init__()
        self.channel = channel
        self.gamma = gamma
        self.b = b
        self.conv = nn.Conv1d(1, 1, kernel_size=self.kernel_size(), padding=(self.kernel_size()-1)//2, bias=False)
        
    def kernel_size(self):
        k = int(abs((math.log2(self.channel) + self.b) / self.gamma))
        return k if k % 2 else k + 1
        
    def forward(self, x):
        # 全局平均池化
        y = x.mean(dim=(2,3), keepdim=True)  # [B,C,1,1]
        # 压缩通道维度
        y = y.squeeze(-1).transpose(-1,-2)  # [B,1,C]
        # 1D卷积
        y = self.conv(y)  # [B,1,C]
        # 激活函数
        y = F.sigmoid(y)  # [B,1,C]
        # 恢复维度
        y = y.transpose(-1,-2).unsqueeze(-1)  # [B,C,1,1]
        # 特征重标定
        return x * y.expand_as(x)