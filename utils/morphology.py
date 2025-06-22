import cv2
import numpy as np

def postprocess_polyp_mask(pred_mask, max_regions=2, min_area_ratio=0.03):
    """
    对息肉预测结果进行形态学后处理
    参数:
        pred_mask: 二值化预测掩码(0和255)
        max_regions: 最大保留区域数
        min_area_ratio: 最小区域面积占比阈值(相对于图像总面积)
    返回:
        处理后的掩码
    """
    # 确保输入是二值图像
    if len(pred_mask.shape) == 3:
        pred_mask = cv2.cvtColor(pred_mask, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(pred_mask, 127, 255, cv2.THRESH_BINARY)
    
    # 计算图像总面积
    total_area = pred_mask.shape[0] * pred_mask.shape[1]
    min_area = total_area * min_area_ratio
    
    # 寻找连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # 如果区域数小于等于1(只有背景)，直接返回
    if num_labels <= 1:
        return pred_mask
    
    # 获取区域面积(跳过背景)
    areas = stats[1:, cv2.CC_STAT_AREA]
    
    # 如果只有一个分割区域，不进行面积过滤
    if num_labels == 2:  # 1背景+1前景
        max_indices = [1]  # 直接保留第一个前景区域
    else:
        # 过滤掉面积太小的区域
        valid_indices = [i+1 for i, area in enumerate(areas) if area >= min_area]
        
        # 如果没有有效区域，返回全黑图像
        if not valid_indices:
            return np.zeros_like(pred_mask, dtype=np.uint8)
        
        # 按面积排序并保留最大的max_regions个区域
        if len(valid_indices) > max_regions:
            valid_areas = [areas[i-1] for i in valid_indices]
            max_indices = [valid_indices[i] for i in np.argsort(valid_areas)[-max_regions:]]
        else:
            max_indices = valid_indices
    
    # 创建新掩码
    new_mask = np.zeros_like(labels, dtype=np.uint8)
    for idx in max_indices:
        new_mask[labels == idx] = 255
    
    # 可选: 进行形态学闭运算填充小孔洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    new_mask = cv2.morphologyEx(new_mask, cv2.MORPH_CLOSE, kernel)
    
    return new_mask