import os
from tqdm import tqdm
from utils.utils_metrics import compute_mIoU, show_results, cal_PR_Curve
import glob
import numpy as np
import torch
import cv2
from model.unet_model_se import R_SEUNet
from model.unet_model_r import ResNet50UNet
from model.unet_model_cbam import R_CBAMUNet
from model.unet_model_eca import R_ECAUNet
from model.unet_model import UNet
from PIL import Image
from utils.morphology import postprocess_polyp_mask



def cal_miou(test_dir="D:/Graduation_Project/Image_Segmentation/data/Test_Images",
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
        # 创建保存灰度图像的目录
        pred_dir = "D:/Graduation_Project/Image_Segmentation/data/results/pred_gray"
        os.makedirs(pred_dir, exist_ok=True)
        # 创建保存轮廓图像的目录
        contour_dir = "D:/Graduation_Project/Image_Segmentation/data/results/pred_record"
        os.makedirs(contour_dir, exist_ok=True)

        print("Load model.")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 加载网络，图片单通道，分类为1。
        net = R_SEUNet(n_channels=1, n_classes=1)
        # 将网络拷贝到deivce中
        net.to(device=device)
        # 加载模型参数
        net.load_state_dict(torch.load('best_model_R_SE_w(45+val+de+AdamW).pth', map_location=device))
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
                        
            #保存概率图像                          
            cv2.imwrite(os.path.join(pred_prob_dir, image_id + ".png"), prob_map_uint8)
            

            # 提取结果
            pred = np.array(pred.data.cpu()[0])[0]
            pred[pred >= 0.5] = 255
            pred[pred < 0.5] = 0
            pred = cv2.resize(pred, (origin_shape[1], origin_shape[0]), interpolation=cv2.INTER_NEAREST)

            # 确保 pred 是 uint8 类型s
            pred = pred.astype(np.uint8)
            # 添加形态学后处理 - 保留最多2个区域，且面积大于1.2%
            pred = postprocess_polyp_mask(pred, max_regions=2, min_area_ratio=0.012)
            
            # 在原始图像上绘制轮廓
            contours, _ = cv2.findContours(pred.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = original_img.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

                   
            # 保存轮廓图像
            cv2.imwrite(os.path.join(contour_dir, image_id + ".jpg"), contour_img)

            # 保存预测结果
            cv2.imwrite(os.path.join(pred_dir, image_id + ".png"), pred)
            

        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get Dice IoU PA Accuracy Precision.")
        print(gt_dir)
        print(pred_dir)
        print(pred_prob_dir)
        print(num_classes)
        print(name_classes)
        hist, IoUs, PA_Recall, Precision, pred_imgs, gt_imgs = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)  
        print("Get Dice IoU PA Accuracy Precision done.")
        miou_out_path = "results/"
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, pred_imgs, gt_imgs)
        cal_PR_Curve(gt_dir, pred_prob_dir, image_ids)  


if __name__ == '__main__':
    cal_miou()
    
