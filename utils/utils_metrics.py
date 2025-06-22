# -*- coding: utf-8 -*-

import csv
import os
from os.path import join
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import precision_recall_curve, average_precision_score
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from tqdm import tqdm



def dice_loss(pred, target, smooth=1e-5):
    #计算Dice loss   
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice  # 返回Dice loss

# 设标签宽W，长H
def fast_hist(label, pred, num_classes):
    mask = (label >= 0) & (label < num_classes)
    hist = np.bincount(
        num_classes * label[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2,
    ).reshape(num_classes, num_classes)
    return hist


def per_class_iu(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    iu[~np.isfinite(iu)] = 0
    return iu


def per_class_PA_Recall(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        pa_recall = np.diag(hist) / hist.sum(1)
    pa_recall[~np.isfinite(pa_recall)] = 0
    return pa_recall


def per_class_Precision(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        precision = np.diag(hist) / hist.sum(0)
    precision[~np.isfinite(precision)] = 0
    return precision


def per_Accuracy(hist):
    with np.errstate(divide='ignore', invalid='ignore'):
        accuracy = np.sum(np.diag(hist)) / hist.sum()
    accuracy = 0 if not np.isfinite(accuracy) else accuracy
    return accuracy

def per_class_dice_scores(hist, name_classes):
    """计算各类别Dice系数并返回"""
    dice_scores = [(2 * np.diag(hist)[i]) / (hist.sum(1)[i] + hist.sum(0)[i] + 1e-6) 
                  for i in range(len(name_classes))]
    return dice_scores

def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes):
    print('Num classes', num_classes)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    gt_imgs = [os.path.join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [os.path.join(pred_dir, x + ".png") for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    skipped_images = 0
    total_dice = 0
    dice_count = 0


    for ind in range(len(gt_imgs)):



        # 读取一张图像分割结果，并转换成numpy数组
        pred = np.array(Image.open(pred_imgs[ind]).convert('L'))  # 确保是单通道图像
        # 调整 pred 图像的大小
        pred = cv2.resize(pred, (512, 512), interpolation=cv2.INTER_NEAREST)

        # 读取一张对应的标签，并转换成numpy数组
        label = np.array(Image.open(gt_imgs[ind]).convert('L'))  # 确保是单通道图像
        # 调整 label 图像的大小
        label = cv2.resize(label, (512, 512), interpolation=cv2.INTER_NEAREST)

        # 确保 label 和 pred 有相同的形状
        if label.shape != pred.shape:
            raise ValueError(f"Label shape {label.shape} does not match pred shape {pred.shape}")

        # 对一张图片计算 21×21 的 hist 矩阵，并累加
        label = np.array([int(x) for x in label.flatten()])
        label[label == 255] = 1

        pred = np.array([int(x) for x in pred.flatten()])
        pred[pred == 255] = 1

        # 确保label和pred是二值图像(0或1)
        label = (label > 0).astype(np.uint8)
        pred = (pred > 0).astype(np.uint8)

        # Debugging: 检查形状和值
        #print(f"Label shape: {label.shape}, Pred shape: {pred.shape}")
        #print(f"Label unique values: {np.unique(label)}, Pred unique values: {np.unique(pred)}")

        # 计算Dice系数
        intersection = np.sum(label * pred)
        union = np.sum(label) + np.sum(pred)
        dice = (2. * intersection) / (union + 1e-6)  # 添加平滑项避免除以0
        dice = np.clip(dice, 0, 1)  # 确保Dice值在0-1范围内
        
        # 打印单个样本的Dice系数
        hist_temp = fast_hist(label, pred, num_classes)
        iou_temp = per_class_iu(hist_temp)
        #print(f"Image {ind + 1}: Dice={dice:.4f}, IoU={iou_temp[1]:.4f}")

        hist += fast_hist(label, pred, num_classes)

        # ------------------------------------------------#
        #   计算所有验证集图片的逐类别mIoU值
        # ------------------------------------------------#
        IoUs = per_class_iu(hist)
        PA_Recall = per_class_PA_Recall(hist)
        Precision = per_class_Precision(hist)

        # 每计算 10 张就输出一下目前已计算的图片中所有类别平均的 mIoU mPA Accuracy mPrecision值
        if ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%; mPrecision-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist),
                100 * np.nanmean(Precision)
            ))

    if skipped_images > 0:
        print(f"Skipped {skipped_images} images due to shape mismatches or errors.")

    
    # 计算平均Dice系数
    dice_scores = per_class_dice_scores(hist, name_classes)
    avg_dice = np.mean(dice_scores)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    for ind_class in range(num_classes):
        print('===> ' + name_classes[ind_class] + ': Dice-' + str(round(dice_scores[ind_class] * 100, 2)) + '%'
              + '; Iou-' + str(round(IoUs[ind_class] * 100, 2)) + '%'
              + '; PA (equal to the Recall)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '%'
              + '; Precision-' + str(round(Precision[ind_class] * 100, 2)) + '%')

    # -----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    # -----------------------------------------------------------------#
    print('===> Average Dice Coefficient: {:.2f}%'.format(avg_dice * 100))
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '%; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '%; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 2)) + '%'
          + '; mPrecision: ' + str(round(np.nanmean(Precision)* 100, 2)) + '%')
    print('mDice/mIoU ratio: {:.2f} (should be close to 2/(1+IoU))'.format(avg_dice / np.nanmean(IoUs)))


    return np.array(hist, int), IoUs, PA_Recall, Precision, pred_imgs, gt_imgs  


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])


def cal_PR_Curve(gt_dir, pred_prob_dir, image_ids):
     # 计算PR曲线
        y_true = []
        y_scores = []
        
        for img_id in image_ids:
            # 确保标签和概率图尺寸匹配
            label = cv2.imread(os.path.join(gt_dir, img_id + ".png"), cv2.IMREAD_GRAYSCALE)
            prob = cv2.imread(os.path.join(pred_prob_dir, img_id + ".png"), cv2.IMREAD_GRAYSCALE)
            
            if label.shape != prob.shape:
                prob = cv2.resize(prob, (label.shape[1], label.shape[0]), 
                                interpolation=cv2.INTER_LINEAR)
            
            y_true.extend((label > 128).astype(int).flatten())
            y_scores.extend((prob / 255.0).flatten())  # 归一化到0-1

        # 计算PR曲线指标
        print(f"Calculating Precision-Recall Curve.")
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
        
        print(f"Save Precision-Recall Curve out to " + pr_curve_path)
        print(f"===> Average Precision: {ap:.4f}")



def show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, pred_imgs, gt_imgs, tick_font_size=12):
    # 计算各类指标
    dice_scores = per_class_dice_scores(hist, name_classes)
    avg_dice = np.mean(dice_scores)  # 计算平均Dice系数
 
    # 通用图表设置
    def create_horizontal_bar_chart(data, title, xlabel, filename):
        plt.figure(figsize=(10, 6))
        bars = plt.barh(name_classes, data, color='#1f77b4')
        plt.title(title, fontsize=tick_font_size+2)
        plt.xlabel(xlabel, fontsize=tick_font_size)
        plt.xlim(0, 1.1)
        plt.grid(axis='x', alpha=0.5)
        
        for bar in bars:
            width = bar.get_width()
            plt.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                   f'{width:.3f}', va='center', fontsize=tick_font_size)
        
        plt.tight_layout()
        path = os.path.join(miou_out_path, filename)
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Save {filename} out to {path}")

    # 生成各类图表
    create_horizontal_bar_chart(
        dice_scores,
        f'Dice Coefficient (Avg = {np.mean(dice_scores)*100:.2f}%)',
        'Dice Score',
        "Dice_Coefficient.png"
    )
    
    create_horizontal_bar_chart(
        IoUs,
        f'mIoU = {np.nanmean(IoUs)*100:.2f}%',
        'IoU Score',
        "mIoU_Score.png"
    )
    
    create_horizontal_bar_chart(
        PA_Recall,
        f'mPA (Recall) = {np.nanmean(PA_Recall)*100:.2f}%',
        'Recall Score',
        "mPA_Recall.png"
    )
    
    create_horizontal_bar_chart(
        Precision,
        f'mPrecision = {np.nanmean(Precision)*100:.2f}%',
        'Precision Score',
        "Precision.png"
    )

    

    # 综合雷达图
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(polar=True)
    
    # 设置雷达图参数
    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)  # 5个指标对应5个角度
    angles = np.concatenate((angles, [angles[0]]))  # 闭合图形
    
    # 准备数据
    data = [
        dice_scores[1],     # 病灶Dice值
        np.mean(IoUs),         # mIoU平均值
        np.mean(PA_Recall),  # Recall平均值
        np.mean(Precision),    # Precision平均值
        per_Accuracy(hist)

    ]
    data = np.concatenate((data, [data[0]]))  # 闭合数据
    
    # 绘制雷达图（平面）
    ax.plot(angles, data, 'o-', linewidth=3, color='#4CAF50')
    ax.fill(angles, data, alpha=0.3, color='#4CAF50')
    
    # 设置雷达图样式
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), 
                     ['Dice', 'mIoU', 'mPA(Recall)', 'mPrecision','Accuracy'],
                     fontsize=tick_font_size+2)
    ax.set_rgrids(np.arange(0, 1.1, 0.2), fontsize=tick_font_size)
    ax.set_ylim(0, 1.1)
    plt.title('Metrics Comparison Radar', fontsize=tick_font_size+4, pad=25)
    
    # 在数据点添加数值标签
    for angle, value in zip(angles[:-1], data[:-1]):
        ax.text(angle, value+0.05, f'{value:.3f}', 
               ha='center', va='center', fontsize=tick_font_size+1, 
               bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7))
    
    # 保存雷达图
    radar_path = os.path.join(miou_out_path, "Metrics_Radar_Chart.png")
    plt.savefig(radar_path, dpi=300, bbox_inches='tight')
    plt.close()
    print("Save Metrics Radar Chart out to " + radar_path)

    

    # 混淆矩阵保存
    with open(os.path.join(miou_out_path, "confusion_matrix.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer_list = []
        writer_list.append([' '] + [str(c) for c in name_classes])
        for i in range(len(hist)):
            writer_list.append([name_classes[i]] + [str(x) for x in hist[i]])
        writer.writerows(writer_list)
    print("Save confusion_matrix out to " + os.path.join(miou_out_path, "confusion_matrix.csv"))
