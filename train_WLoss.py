import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

from model.unet_model_se import R_SEUNet
from utils.dataset import ISBI_Loader
from utils.utils_metrics import dice_loss  # 如果你还用作对比指标可保留


# === 自定义置信度加权损失函数 ===
class ConfidenceWeightedLoss(nn.Module):
    def __init__(self, alpha=1.0, lambda_weight=0.5, eps=1e-7):
        """
        :param alpha: 控制置信度加权的强度（alpha 越大，对不确定区域的关注越强）
        :param lambda_weight: 控制 BCE 和 Dice 损失的组合比例
        :param eps: 平滑项，防止除零
        """
        super(ConfidenceWeightedLoss, self).__init__()
        self.alpha = alpha
        self.lambda_weight = lambda_weight
        self.eps = eps

    def forward(self, preds, targets):
        """
        :param preds: 模型输出的概率图 (batch, 1, H, W)
        :param targets: 对应的标签图 (batch, 1, H, W)
        :return: 混合加权损失值
        """
        probs = torch.sigmoid(preds)
        conf_weight = 1 + self.alpha * (1 - torch.abs(2 * probs - 1))

        # Weighted BCE
        bce_loss = - (targets * torch.log(probs + self.eps) + (1 - targets) * torch.log(1 - probs + self.eps))
        weighted_bce = (conf_weight * bce_loss).mean()

        # Weighted Dice
        intersection = torch.sum(conf_weight * probs * targets)
        union = torch.sum(conf_weight * probs) + torch.sum(conf_weight * targets)
        weighted_dice = 1 - (2.0 * intersection + self.eps) / (union + self.eps)

        return self.lambda_weight * weighted_bce + (1 - self.lambda_weight) * weighted_dice


# === 主训练函数 ===
def train_net(net, device, data_path, epochs=100, batch_size=4, lr=3e-4):
    train_dataset = ISBI_Loader(data_path, mode='train')
    val_dataset = ISBI_Loader(data_path, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)

    # 使用新的损失函数
    loss_fn = ConfidenceWeightedLoss(alpha=1.0, lambda_weight=0.5)

    train_losses = []
    val_losses = []
    best_loss = float('inf')

    with tqdm(total=epochs * len(train_loader)) as pbar:
        for epoch in range(epochs):
            net.train()
            epoch_train_loss = 0.0

            for image, label in train_loader:
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)

                optimizer.zero_grad()
                pred = net(image)

                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()

                epoch_train_loss += loss.item() * image.size(0)
                pbar.update(1)

            epoch_train_loss /= len(train_dataset)
            train_losses.append(epoch_train_loss)

            # 验证
            net.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for image, label in val_loader:
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    pred = net(image)

                    loss = loss_fn(pred, label)
                    epoch_val_loss += loss.item() * image.size(0)

            epoch_val_loss /= len(val_dataset)
            val_losses.append(epoch_val_loss)

            scheduler.step(epoch_val_loss)

            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                torch.save(net.state_dict(), 'best_model_R_se_w(+val+de+AdamW).pth')
                print(f'New best model saved with Val Loss: {best_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.2e}')

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    # 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, 'r-', label='Validating Loss')
    best_epoch = val_losses.index(min(val_losses)) + 1
    plt.scatter(best_epoch, train_losses[best_epoch-1], c='blue', s=100)
    plt.scatter(best_epoch, val_losses[best_epoch-1], c='red', s=100)
    plt.text(best_epoch, (train_losses[best_epoch-1] + val_losses[best_epoch-1])/2,
             f'Epoch {best_epoch}\nTrain: {train_losses[best_epoch-1]:.4f}\nVal: {val_losses[best_epoch-1]:.4f}',
             ha='center', va='bottom', fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))
    plt.title('Training & Validation Loss Curve (Confidence Weighted)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/train_val_loss_weighted.png')
    plt.close()

    print(
        f'===> Best Epoch: {best_epoch}, Train Loss: {train_losses[best_epoch - 1]:.4f}, '
        f'Val Loss: {val_losses[best_epoch - 1]:.4f}'
    )


# === 主入口 ===
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = R_SEUNet(n_channels=1, n_classes=1)
    net.to(device=device)
    data_path = "D:/Graduation_Project/Image_Segmentation/data"
    # 模型参数打印
    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params:,}")
    print("Training and evaluating, please wait.")
    train_net(net, device, data_path, epochs=50, batch_size=4)
    print("Training Complete.")
