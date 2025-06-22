# 修改导入
from model.unet_model_se import R_SEUNet
from utils.dataset import ISBI_Loader
from torch import optim
import torch.nn as nn
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from utils.utils_metrics import dice_loss  
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_net(net, device, data_path, epochs=100, batch_size=4, lr=3e-4):
    # 加载训练集和验证集
    train_dataset = ISBI_Loader(data_path, mode='train')
    val_dataset = ISBI_Loader(data_path, mode='val')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                            batch_size=1,
                                            shuffle=False)

     
    # 定义优化器
    optimizer = optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
    # 添加学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    criterion = nn.BCEWithLogitsLoss()  # 交叉熵损失
    alpha = 0.5  # 交叉熵和Dice loss的混合权重
    
    # 记录训练和验证loss
    train_losses = []
    val_losses = []
    best_loss = float('inf')

    with tqdm(total=epochs*len(train_loader)) as pbar:
        for epoch in range(epochs):
            net.train()
            epoch_train_loss = 0.0
            
            # 训练阶段
            for image, label in train_loader:
                optimizer.zero_grad()
                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.float32)
                pred = net(image)
                
                # 计算混合loss
                ce_loss = criterion(pred, label)
                dc_loss = dice_loss(torch.sigmoid(pred), label)
                loss = alpha * ce_loss + (1 - alpha) * dc_loss
                
                loss.backward()
                optimizer.step()
                epoch_train_loss += loss.item() * image.size(0)
                pbar.update(1)
            
            # 计算平均训练loss
            epoch_train_loss /= len(train_dataset)
            train_losses.append(epoch_train_loss)
            
            # 验证阶段
            net.eval()
            epoch_val_loss = 0.0
            with torch.no_grad():
                for image, label in val_loader:
                    image = image.to(device=device, dtype=torch.float32)
                    label = label.to(device=device, dtype=torch.float32)
                    pred = net(image)
                    
                    # 计算混合loss
                    ce_loss = criterion(pred, label)
                    dc_loss = dice_loss(torch.sigmoid(pred), label)
                    loss = alpha * ce_loss + (1 - alpha) * dc_loss

                    epoch_val_loss += loss.item() * image.size(0)
            
            # 计算平均验证loss
            epoch_val_loss /= len(val_dataset)
            val_losses.append(epoch_val_loss)
            
            # 更新学习率
            scheduler.step(epoch_val_loss)

            # 保存最佳模型
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                torch.save(net.state_dict(), 'best_model_R_SE(+val+de+AdamW).pth')
                print(f'New best model saved with Val Loss: {best_loss:.4f}, Current LR: {optimizer.param_groups[0]["lr"]:.2e}')

            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}')

    # 绘制训练和验证loss曲线
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, 'r-', label='Validating Loss')
    
    # 标注最佳epoch点
    best_epoch = val_losses.index(min(val_losses)) + 1
    plt.scatter(best_epoch, train_losses[best_epoch-1], c='blue', s=100)
    plt.scatter(best_epoch, val_losses[best_epoch-1], c='red', s=100)
    
    # 添加文本标注
    plt.text(best_epoch, (train_losses[best_epoch-1] + val_losses[best_epoch-1])/2,
             f'Epoch {best_epoch}\nBest Train: {train_losses[best_epoch-1]:.4f}\nBest Val: {val_losses[best_epoch-1]:.4f}',
             ha='center', va='bottom', fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.title('Training and Validating Loss Curves(ce)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    os.makedirs('results', exist_ok=True)
    plt.savefig('results/train_val_loss_curve(+de).png')
    plt.close()

    # 输出最佳epoch
    print(
        f'===> Best Epoch: {best_epoch}, Train Loss: {train_losses[best_epoch - 1]:.4f}, '
        f'Val Loss: {min(val_losses):.4f}')

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
