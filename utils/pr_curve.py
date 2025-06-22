import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import glob
import numpy as np
import torch
import cv2
from model.unet_model import ResNet50UNet
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def cal_PR_Curve(test_dir="D:/Graduation_Project/Image_Segmentation/data/Test_Images",
             gt_dir="D:/Graduation_Project/Image_Segmentation/data/Test_Labels"):
    # ---------------------------------------------------------------------------#
    #   miou_mode用于指定该文件运行时计算的内容
    #   miou_mode为0代表整个miou计算流程，包括获得预测结果、计算miou。
    #   miou_mode为1代表仅仅获得预测结果。
    #   miou_mode为2代表仅仅计算miou。
    # ---------------------------------------------------------------------------#
    miou_mode = 0
    # ------------------------------#
    #   分类个数+1、如2+1
    # ------------------------------#
    num_classes = 2
    # --------------------------------------------#
    #   区分的种类，和json_to_dataset里面的一样
    # --------------------------------------------#
    name_classes = ["background", "polyp"]
    # name_classes    = ["_background_","cat","dog"]
    # -------------------------------------------------------#
    #   指向VOC数据集所在的文件夹
    #   默认指向根目录下的VOC数据集
    # -------------------------------------------------------#
    # 计算结果和gt的结果进行比对

    # 加载模型

    if miou_mode == 0 or miou_mode == 1:

        #创建保存概率图像的目录
        pred_prob_dir = "D:/Graduation_Project/Image_Segmentation/data/results/pred_prob"
        os.makedirs(pred_prob_dir, exist_ok=True)
        
        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        net = ResNet50UNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model_ResNet50.pth', map_location=device))
        # 测试模式
        net.eval()
        print("Load model done.")

        img_names = os.listdir(test_dir)
        image_ids = [image_name.split(".")[0] for image_name in img_names]

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path = os.path.join(test_dir, image_id + ".jpg")
            original_img = cv2.imread(image_path)  # 保留原始图像
            origin_shape = original_img.shape
            # 转为灰度图
            img = cv2.cvtColor(original_img, cv2.COLOR_RGB2GRAY)  # 修改这里，使用original_img作为输入
            img = cv2.resize(img, (512, 512))
            # 转为batch为1，通道为1，大小为512*512的数组
            img = img.reshape(1, 1, img.shape[0], img.shape[1])
            # 转为tensor
            img_tensor = torch.from_numpy(img)
            # 将tensor拷贝到device中，只用cpu就是拷贝到cpu中，用cuda就是拷贝到cuda中。
            img_tensor = img_tensor.to(device=device, dtype=torch.float32)
            # 预测
            pred = net(img_tensor)
            
            # 提取概率图（使用sigmoid确保0-1范围）
            prob_map = torch.sigmoid(pred).detach().cpu().numpy()[0,0]  # 添加detach()
            
            # 检查概率图数值范围
            #print(f"概率图范围: min={prob_map.min():.4f}, max={prob_map.max():.4f}")
            
            # 保存概率图（0-255范围）
            prob_map_uint8 = (prob_map * 255).astype(np.uint8)
            prob_map_uint8 = cv2.resize(prob_map_uint8, (origin_shape[1], origin_shape[0]), 
                                      interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(pred_prob_dir, image_id + ".png"), prob_map_uint8)
        print("Get predict result done.")

        # 计算PR曲线
        y_true = []
        y_scores = []
        
        for img_id in image_ids:
            # 确保标签和概率图尺寸匹配
            label = cv2.imread(os.path.join(gt_dir, img_id + ".png"), cv2.IMREAD_GRAYSCALE)
            prob = cv2.imread(os.path.join(pred_dir, img_id + ".png"), cv2.IMREAD_GRAYSCALE)
            
            if label.shape != prob.shape:
                prob = cv2.resize(prob, (label.shape[1], label.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
            
            y_true.extend((label > 128).astype(int).flatten())
            y_scores.extend((prob / 255.0).flatten())  # 归一化到0-1

        # 计算PR曲线指标
        print(f"Calculating PR Curve.")
        with tqdm(total=2) as pbar:
            # 计算PR曲线指标
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pbar.update(1)
            ap = average_precision_score(y_true, y_scores)
            pbar.update(1)
        
        
        
        # 绘制PR曲线
        plt.figure(figsize=(10, 6))
        plt.plot(recall, precision, label=f'AP={ap:.2f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        pr_path = "results/"
        pr_curve_path = os.path.join(pr_path, 'PR_Curve.png')
        plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Save Precision-Recall Curve out to" + pr_curve_path)
        print(f"===>Average Precision: {ap:.4f}")

if __name__ == '__main__':
    cal_PR_Curve()