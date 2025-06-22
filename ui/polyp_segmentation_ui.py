import sys
import os
# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import psutil
from PyQt5.QtWidgets import QAction, QMessageBox
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QSlider, QComboBox, QPushButton, QFileDialog, 
                            QStatusBar, QProgressBar, QGroupBox, QScrollArea, QGridLayout, QSizePolicy)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
import cv2
import glob
import numpy as np
from model.unet_model_se import R_SEUNet
from model.unet_model_r import ResNet50UNet
from model.unet_model_cbam import R_CBAMUNet
from model.unet_model_eca import R_ECAUNet
from model.unet_model import UNet
import torch
from utils.morphology import postprocess_polyp_mask

try:
    from resource_loader import get_resource_path
except ImportError:
    from .resource_loader import get_resource_path

class PolypSegmentationUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("胃镜图像息肉分割系统")
        self.setGeometry(100, 100, 1200, 1000)
        
        # 初始化模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = None  # 初始化为None，将在process_image中根据选择初始化
                
        # 主界面布局
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout = QVBoxLayout(self.main_widget)
        
       # 创建系统菜单
        menubar = self.menuBar()
        
        # 添加"文件"菜单
        file_menu = menubar.addMenu("文件(&F)")
        
        # 添加加载图像菜单项
        load_action = QAction("加载图像", self)
        load_action.setShortcut("Ctrl+O")
        load_action.triggered.connect(self.load_image)
        file_menu.addAction(load_action)

        # 添加保存结果菜单项
        save_action = QAction("保存结果", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)

        # 添加"开始"菜单
        process_menu = menubar.addMenu("开始(&P)")
        
        # 添加开始处理菜单项
        process_action = QAction("开始处理", self)
        process_action.setShortcut("Ctrl+D")
        process_action.triggered.connect(lambda: self.process_image(show_progress=True))
        process_menu.addAction(process_action)
        
        # 添加批量处理菜单项
        batch_action = QAction("批量处理", self)
        batch_action.setShortcut("Ctrl+F")
        batch_action.triggered.connect(self.batch_process)
        process_menu.addAction(batch_action)
        
        
        
        # 添加"查看"菜单
        view_menu = menubar.addMenu("查看(&V)")

        # 添加清除菜单项
        clear_action = QAction("清除图像", self)
        clear_action.setShortcut("Ctrl+X")
        clear_action.triggered.connect(self.clear_images)
        view_menu.addAction(clear_action)

        # 添加主题设置菜单项
        theme_menu = view_menu.addMenu("主题设置")
        
        # 白天模式
        light_action = QAction("白天模式", self)
        light_action.setShortcut("Ctrl+L")
        light_action.triggered.connect(self.set_light_theme)
        theme_menu.addAction(light_action)
        
        # 夜间模式
        dark_action = QAction("夜间模式", self)
        dark_action.setShortcut("Ctrl+B")
        dark_action.triggered.connect(self.set_dark_theme)
        theme_menu.addAction(dark_action)


        # 添加"帮助"菜单
        help_menu = menubar.addMenu("帮助(&H)")
        
        # 添加关于菜单项
        about_action = QAction("关于", self)
        about_action.setShortcut("Ctrl+A")
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
        
        # 创建上部网格布局 (4个区域)
        self.grid_layout = QGridLayout()
        
        # 左上区域 - 原始图像 (添加滑动条)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("background-color: transparent;")
        self.original_scroll = QScrollArea()
        self.original_scroll.setWidget(self.original_label)
        self.original_scroll.setWidgetResizable(True)

        # 创建滑动条和页数控制 (移动到图像框上方)
        slider_control = QHBoxLayout()
        
        # 添加减号按钮
        self.prev_btn = QPushButton("-")
        self.prev_btn.setFixedWidth(30)
        self.prev_btn.clicked.connect(self.prev_image)
        slider_control.addWidget(self.prev_btn)
        
        # 添加滑动条
        self.image_slider = QSlider(Qt.Horizontal)
        self.image_slider.setRange(0, 0)
        self.image_slider.valueChanged.connect(self.slider_changed)
        slider_control.addWidget(self.image_slider)
        
        # 添加加号按钮
        self.next_btn = QPushButton("+")
        self.next_btn.setFixedWidth(30)
        self.next_btn.clicked.connect(self.next_image)
        slider_control.addWidget(self.next_btn)
        
        # 添加页数显示
        self.page_label = QLabel("0/0")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setFixedWidth(80)
        slider_control.addWidget(self.page_label)

        # 修改网格布局结构
        self.grid_layout.addWidget(QLabel("原始图像"), 0, 0)
        self.grid_layout.addLayout(slider_control, 5, 0)  
        self.grid_layout.addWidget(self.original_scroll, 1, 0) 
        
        # 右上区域 - 参数设置
        self.params_group = QGroupBox("参数设置")
        self.params_layout = QVBoxLayout(self.params_group)
        
        # 置信度阈值
        self.threshold_label = QLabel("置信度阈值: 0.5")
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(50)
        self.threshold_slider.valueChanged.connect(self.update_threshold)
        
        # 算法选择
        self.algorithm_label = QLabel("分割算法:")
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItems(["R_SE-UNet(推荐)", "R_CBAM-UNet(推荐)", "R_ECA-UNet", "ResNet50-UNet", "UNet(不推荐)"])
        
        # 输出格式
        self.format_label = QLabel("输出格式:")
        self.format_combo = QComboBox()
        self.format_combo.addItems(["PNG", "JPG", "BMP", "TIFF"])
                
        # 加载图像按钮
        self.load_btn = QPushButton("加载图像")
        self.load_btn.clicked.connect(self.load_image)
                
        # 处理按钮
        self.process_btn = QPushButton("开始处理")
        self.process_btn.clicked.connect(lambda: self.process_image(show_progress=True))
        
        # 批量处理按钮
        self.batch_btn = QPushButton("批量处理")
        self.batch_btn.clicked.connect(self.batch_process)

        # 保存按钮
        self.save_btn = QPushButton("保存结果")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        
        # 添加到参数布局
        self.params_layout.addWidget(self.threshold_label)
        self.params_layout.addWidget(self.threshold_slider)
        self.params_layout.addWidget(self.algorithm_label)
        self.params_layout.addWidget(self.algorithm_combo)
        self.params_layout.addWidget(self.format_label)
        self.params_layout.addWidget(self.format_combo)
        self.params_layout.addWidget(self.load_btn)
        self.params_layout.addWidget(self.process_btn)
        self.params_layout.addWidget(self.batch_btn)
        self.params_layout.addWidget(self.save_btn)
        self.params_layout.addStretch()
        
        self.grid_layout.addWidget(self.params_group, 0, 1, 2, 1)
        
        # 左下区域 - 轮廓图像
        self.contour_label = QLabel()
        self.contour_label.setAlignment(Qt.AlignCenter)
        self.contour_label.setStyleSheet("background-color: transparent;")
        self.contour_scroll = QScrollArea()
        self.contour_scroll.setWidget(self.contour_label)
        self.contour_scroll.setWidgetResizable(True)
        self.grid_layout.addWidget(QLabel("分割轮廓图像"), 2, 0)
        self.grid_layout.addWidget(self.contour_scroll, 3, 0)
        
        # 右下区域 - 分割灰度图像
        self.segmented_label = QLabel()
        self.segmented_label.setAlignment(Qt.AlignCenter)
        self.segmented_label.setStyleSheet("background-color: transparent;")
        self.segmented_scroll = QScrollArea()
        self.segmented_scroll.setWidget(self.segmented_label)
        self.segmented_scroll.setWidgetResizable(True)
        self.grid_layout.addWidget(QLabel("分割灰度图像"), 2, 1)
        self.grid_layout.addWidget(self.segmented_scroll, 3, 1)
        
        # 设置网格布局的伸缩因子
        self.grid_layout.setRowStretch(1, 1)
        self.grid_layout.setRowStretch(3, 1)
        self.grid_layout.setColumnStretch(0, 1)
        self.grid_layout.setColumnStretch(1, 1)
        
        self.main_layout.addLayout(self.grid_layout)
        
        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # 进度条
        progress_layout = QHBoxLayout()
        progress_label = QLabel("处理进度:")
        progress_layout.addWidget(progress_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        progress_layout.addWidget(self.progress_bar)
        
        progress_widget = QWidget()
        progress_widget.setLayout(progress_layout)
        self.status_bar.addPermanentWidget(progress_widget)
        
        # 内存使用
        self.memory_label = QLabel()
        self.update_memory_usage()
        self.status_bar.addPermanentWidget(self.memory_label)
        
        # 定时器更新内存使用
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_memory_usage)
        self.timer.start(1000)
        
        # 初始化变量
        self.current_image = None
        self.original_image = None
        self.segmented_image = None
        self.contour_image = None
        self.threshold = 0.5
        self.processing_time = 0
        self.image_files = []
        self.current_index = 0
        self.batch_results = []
        
        # 启用拖放功能
        self.setAcceptDrops(True)
        self.original_label.setAcceptDrops(True)
        
        # 启用窗口大小变化事件
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    
    def update_memory_usage(self):
        memory = psutil.virtual_memory()
        self.memory_label.setText(f"内存: {memory.percent}%")
    
    def update_threshold(self, value):
        self.threshold = value / 100
        self.threshold_label.setText(f"置信度阈值: {self.threshold:.2f}")
        if self.segmented_image is not None:
            self.display_results()
    
  
    
    
    def load_image(self):
        """加载图像按钮点击事件处理函数"""
        # 弹出文件选择对话框
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择胃镜图像", "", 
            "图像文件 (*.png *.jpg *.jpeg *.bmp *.tiff)"
        )
        if not file_path:  # 用户取消选择
            return
        
        try:
            # 清除批量处理数据
            self.image_files = []
            self.batch_results = []
            self.current_index = 0
            self.progress_bar.setValue(0)
            self.image_slider.setRange(0, 0)
            self.page_label.setText("1/1")
            # 读取图像文件
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("无法读取图像文件")
                
            # 复制原始图像用于显示
            self.current_image = self.original_image.copy()
            # 显示原始图像到界面
            self.display_image(self.current_image)
            # 清空分割灰度图像和轮廓图像显示区域
            self.segmented_image = None  # 清除分割结果记录
            self.contour_image = None    # 清除轮廓结果记录
            self.segmented_label.setPixmap(QPixmap())  # 清除分割标签显示
            self.contour_label.setPixmap(QPixmap())    # 清除轮廓标签显示
            self.segmented_label.setStyleSheet("background-color: transparent;")  # 恢复透明背景
            self.contour_label.setStyleSheet("background-color: transparent;")    # 恢复透明背景
            # 更新状态栏
            self.status_bar.showMessage(f"已加载: {os.path.basename(file_path)}")
            # 禁用保存按钮（未处理时不可保存）
            self.save_btn.setEnabled(False)
            
        except Exception as e:
            self.status_bar.showMessage(f"加载错误: {str(e)}")
    
    def display_image(self, image):
        if image is None:
            return
            
        # 强制更新所有滚动区域几何属性
        self.original_scroll.viewport().updateGeometry()
        self.segmented_scroll.viewport().updateGeometry()
        self.contour_scroll.viewport().updateGeometry()
        QApplication.processEvents()
        
        # 获取最新的视口大小
        original_size = self.original_scroll.viewport().size()
        segmented_size = self.segmented_scroll.viewport().size()
        contour_size = self.contour_scroll.viewport().size()

        # 显示原始图像（保持宽高比）
        h, w, ch = image.shape
        bytes_per_line = ch * w
        q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        
        # 获取ScrollArea的视口(viewport)大小作为参考尺寸
        viewport_size = self.original_scroll.viewport().size()
        scaled_pixmap = pixmap.scaled(
            viewport_size,
            Qt.KeepAspectRatio, 
            Qt.SmoothTransformation
        )
        self.original_label.setPixmap(scaled_pixmap)
        self.original_label.resize(scaled_pixmap.size())

        # 如果有分割结果，显示分割图像和轮廓图像
        if self.segmented_image is not None:
            # 显示分割灰度图像（保持宽高比）
            seg_h, seg_w = self.segmented_image.shape[:2]
            seg_q_img = QImage(self.segmented_image.data, seg_w, seg_h, seg_w, QImage.Format_Grayscale8)
            seg_pixmap = QPixmap.fromImage(seg_q_img)
            scaled_seg_pixmap = seg_pixmap.scaled(
            self.segmented_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
            )
            self.segmented_label.setPixmap(scaled_seg_pixmap)
            self.segmented_label.resize(scaled_seg_pixmap.size())
            
            # 显示轮廓图像（保持宽高比）
            contour_h, contour_w, _ = self.contour_image.shape
            contour_q_img = QImage(self.contour_image.data, contour_w, contour_h, 
                             contour_w * 3, QImage.Format_RGB888).rgbSwapped()
            contour_pixmap = QPixmap.fromImage(contour_q_img)
            scaled_contour_pixmap = contour_pixmap.scaled(
            self.contour_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
            )
            self.contour_label.setPixmap(scaled_contour_pixmap)
            self.contour_label.resize(scaled_contour_pixmap.size())

    def process_image(self, show_progress=True):
        if self.original_image is None:
            self.status_bar.showMessage("错误: 请先加载图像")
            return
            
        try:
            start_time = time.time()
            if show_progress:
                self.progress_bar.setValue(0)
                QApplication.processEvents()  # 强制UI立即更新
            
            # 根据选择的算法初始化模型
            algorithm = self.algorithm_combo.currentText()
            if algorithm == "R_SE-UNet(推荐)":
                model_path = get_resource_path('best_model_R_SE_w(45+val+de+AdamW).pth')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件未找到: {model_path}")
                self.net = R_SEUNet(n_channels=1, n_classes=1).to(self.device)
                
            elif algorithm == "ResNet50-UNet":
                model_path = get_resource_path('best_model_ResNet50_w(44+val+de+AdamW).pth')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件未找到: {model_path}")
                self.net = ResNet50UNet(n_channels=1, n_classes=1).to(self.device)
                
            elif algorithm == "R_CBAM-UNet(推荐)":
                model_path = get_resource_path('best_model_R_CBAM_w(49+val+de+AdamW).pth')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件未找到: {model_path}")
                self.net = R_CBAMUNet(n_channels=1, n_classes=1).to(self.device)
                
            elif algorithm == "R_ECA-UNet":
                model_path = get_resource_path('best_model_R_ECA_w(44+val+de+AdamW).pth')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件未找到: {model_path}")
                self.net = R_ECAUNet(n_channels=1, n_classes=1).to(self.device)
            
            elif algorithm == "UNet(不推荐)":
                model_path = get_resource_path('best_model_w(48+val+de+AdamW).pth')
                if not os.path.exists(model_path):
                    raise FileNotFoundError(f"模型文件未找到: {model_path}")
                self.net = UNet(n_channels=1, n_classes=1).to(self.device)    
            
             # 加载模型权重
            if self.net is not None:
                self.net.load_state_dict(torch.load(model_path, map_location=self.device))
                self.net.eval()
                if show_progress:
                    self.progress_bar.setValue(40)  # 模型加载完成
                    QApplication.processEvents()
                    
            # 预处理图像
            gray_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(gray_img, (512, 512))
            img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0).to(self.device)
            if show_progress:
                self.progress_bar.setValue(60)  # 预处理完成
                QApplication.processEvents()
                
            # 预测
            with torch.no_grad():
                pred = self.net(img_tensor)
                prob_map = torch.sigmoid(pred).cpu().numpy()[0,0]
                if show_progress:
                    self.progress_bar.setValue(80)  # 预测完成
                    QApplication.processEvents()
            
            # 生成分割结果
            prob_map = cv2.resize(prob_map, (self.original_image.shape[1], self.original_image.shape[0]))
            binary_mask = (prob_map > self.threshold).astype(np.uint8) * 255
            
            # 添加形态学后处理
            processed_mask = postprocess_polyp_mask(binary_mask, max_regions=2, min_area_ratio=0.012)
            if show_progress:
                self.progress_bar.setValue(90)
                
            # 生成轮廓图像
            contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = self.original_image.copy()
            cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
            
            # 保存结果
            self.segmented_image = processed_mask
            self.contour_image = contour_img
            if show_progress:
                self.progress_bar.setValue(100)
                
            # 显示结果
            self.display_results()
            
            # 更新状态
            self.processing_time = time.time() - start_time
            self.status_bar.showMessage(f"处理完成 - 耗时: {self.processing_time:.2f}秒")
            self.save_btn.setEnabled(True)
            
        except Exception as e:
            self.status_bar.showMessage(f"处理错误: {str(e)}")
            if show_progress:
                self.progress_bar.setValue(0)

    def resizeEvent(self, event):
        """窗口大小变化事件处理"""
        super().resizeEvent(event)
        # 重新显示当前图像以适配新的大小
        if self.original_image is not None:
            self.display_image(self.current_image)
        if self.segmented_image is not None:
            self.display_results()

    def display_results(self):
        if self.original_image is None or self.segmented_image is None:
            return

        # 确保获取最新的视口大小
        viewport_size = self.original_scroll.viewport().size()

        # 显示原始图像（保持宽高比）
        h, w, ch = self.original_image.shape
        bytes_per_line = ch * w
        q_img = QImage(self.original_image.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(q_img)
        scaled_pixmap = pixmap.scaled(
            self.original_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.original_label.setPixmap(scaled_pixmap)
        self.original_label.resize(scaled_pixmap.size())

        # 显示分割灰度图像（保持宽高比）
        seg_h, seg_w = self.segmented_image.shape[:2]
        seg_q_img = QImage(self.segmented_image.data, seg_w, seg_h, seg_w, QImage.Format_Grayscale8)
        seg_pixmap = QPixmap.fromImage(seg_q_img)
        scaled_seg_pixmap = seg_pixmap.scaled(
            self.segmented_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.segmented_label.setPixmap(scaled_seg_pixmap)
        self.segmented_label.resize(scaled_seg_pixmap.size())

        # 显示轮廓图像（保持宽高比）
        contour_h, contour_w, _ = self.contour_image.shape
        contour_q_img = QImage(self.contour_image.data, contour_w, contour_h, 
                             contour_w * 3, QImage.Format_RGB888).rgbSwapped()
        contour_pixmap = QPixmap.fromImage(contour_q_img)
        scaled_contour_pixmap = contour_pixmap.scaled(
            self.contour_scroll.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.contour_label.setPixmap(scaled_contour_pixmap)
        self.contour_label.resize(scaled_contour_pixmap.size())

    def save_results(self):
        if not self.batch_results and self.segmented_image is None:
            self.status_bar.showMessage("错误: 没有可保存的结果")
            return
            
        try:
            format = self.format_combo.currentText().lower()
            
            # 弹出保存文件夹对话框
            save_dir = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
            if not save_dir:
                return
                
            # 批量保存所有结果
            if self.batch_results:
                for i, result in enumerate(self.batch_results):
                    base_name = os.path.splitext(os.path.basename(self.image_files[i]))[0]
                    
                    # 保存分割结果
                    seg_path = os.path.join(save_dir, f"{base_name}_seg.{format}")
                    if format == 'png':
                        cv2.imwrite(seg_path, result['segmented'])
                    elif format == 'tiff':
                        cv2.imwrite(seg_path, result['segmented'])
                    elif format == 'jpg' or format == 'jpeg':
                        cv2.imwrite(seg_path, result['segmented'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    elif format == 'bmp':
                        cv2.imwrite(seg_path, result['segmented'])
                        
                    # 保存轮廓结果
                    contour_path = os.path.join(save_dir, f"{base_name}_contour.{format}")
                    if format == 'png':
                        cv2.imwrite(contour_path, result['contour'])
                    elif format == 'tiff':
                        cv2.imwrite(contour_path, result['contour'])
                    elif format == 'jpg' or format == 'jpeg':
                        cv2.imwrite(contour_path, result['contour'], [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    elif format == 'bmp':
                        cv2.imwrite(contour_path, result['contour'])
                
                self.status_bar.showMessage(f"批量结果已保存到: {save_dir}")
            # 单张保存
            else:
                base_name = os.path.splitext(os.path.basename(self.status_bar.currentMessage().split(":")[-1].strip()))[0]
                
                # 保存分割结果
                seg_path = os.path.join(save_dir, f"{base_name}_seg.{format}")
                if format == 'png':
                    cv2.imwrite(seg_path, self.segmented_image)
                elif format == 'tiff':
                    cv2.imwrite(seg_path, self.segmented_image)
                elif format == 'jpg' or format == 'jpeg':
                    cv2.imwrite(seg_path, self.segmented_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                elif format == 'bmp':
                    cv2.imwrite(seg_path, self.segmented_image)
                    
                # 保存轮廓结果
                contour_path = os.path.join(save_dir, f"{base_name}_contour.{format}")
                if format == 'png':
                    cv2.imwrite(contour_path, self.contour_image)
                elif format == 'tiff':
                    cv2.imwrite(contour_path, self.contour_image)
                elif format == 'jpg' or format == 'jpeg':
                    cv2.imwrite(contour_path, self.contour_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                elif format == 'bmp':
                    cv2.imwrite(contour_path, self.contour_image)
                    
                self.status_bar.showMessage(f"结果已保存到: {save_dir}")
            
        except Exception as e:
            self.status_bar.showMessage(f"保存错误: {str(e)}")

        
    
    def dragEnterEvent(self, event):
        """拖拽进入事件处理"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        """拖放事件处理"""
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                try:
                    # 清除批量处理数据
                    self.image_files = []
                    self.batch_results = []
                    self.current_index = 0
                    self.progress_bar.setValue(0)
                    self.image_slider.setRange(0, 0)
                    self.page_label.setText("1/1")
                    
                    # 读取图像文件
                    self.original_image = cv2.imread(file_path)
                    if self.original_image is None:
                        raise ValueError("无法读取图像文件")
                        
                    # 复制原始图像用于显示
                    self.current_image = self.original_image.copy()
                    # 显示原始图像到界面
                    self.display_image(self.current_image)
                    # 清空分割结果
                    self.segmented_image = None
                    self.contour_image = None
                    self.segmented_label.setPixmap(QPixmap())
                    self.contour_label.setPixmap(QPixmap())
                    # 更新状态栏
                    self.status_bar.showMessage(f"已加载: {os.path.basename(file_path)}")
                    # 禁用保存按钮
                    self.save_btn.setEnabled(False)
                    break
                except Exception as e:
                    self.status_bar.showMessage(f"加载错误: {str(e)}")

    def batch_process(self):
        """批量处理文件夹中的图像"""
        folder_path = QFileDialog.getExistingDirectory(self, "选择图像文件夹")
        if not folder_path:
            return
            
        # 获取文件夹中所有支持的图像文件
        self.image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']:
            self.image_files.extend(glob.glob(os.path.join(folder_path, ext)))
            
        if not self.image_files:
            self.status_bar.showMessage("错误: 文件夹中没有支持的图像文件")
            return
            
        self.current_index = 0
        self.batch_results = []
        
        # 禁用处理按钮，防止重复处理
        self.process_btn.setEnabled(False)
        self.status_bar.showMessage("正在批量处理图像...")
        
        # 处理所有图像
        for i, file_path in enumerate(self.image_files):
            try:
                # 加载图像
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("无法读取图像文件")
                    
                # 处理图像（不显示单张处理进度）
                self.process_image(show_progress=False)
                
                # 保存结果
                self.batch_results.append({
                    'segmented': self.segmented_image,
                    'contour': self.contour_image
                })
                
                # 更新整体进度（基于已处理图像数量）
                self.progress_bar.setValue(int((i + 1) / len(self.image_files) * 100))
                QApplication.processEvents()  # 更新UI
                
            except Exception as e:
                self.status_bar.showMessage(f"处理 {os.path.basename(file_path)} 时出错: {str(e)}")
                continue
        # 处理完成后更新滑动条和页数显示
        self.image_slider.setRange(0, len(self.image_files)-1)
        self.image_slider.setValue(0)
        self.page_label.setText(f"1/{len(self.image_files)}")        
        # 处理完成后显示第一张图像
        self.current_index = 0
        self.image_slider.setRange(0, len(self.image_files)-1)
        self.image_slider.setValue(0)
        self.load_current_image()
        self.process_btn.setEnabled(True)
        self.status_bar.showMessage("批量处理完成")

    def load_current_image(self):
        """加载当前索引的图像"""
        if not self.image_files:
            return
            
        file_path = self.image_files[self.current_index]
        try:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                raise ValueError("无法读取图像文件")
                
            self.current_image = self.original_image.copy()
            self.display_image(self.current_image)
            
            # 如果有处理结果则显示
            if self.current_index < len(self.batch_results):
                self.segmented_image = self.batch_results[self.current_index]['segmented']
                self.contour_image = self.batch_results[self.current_index]['contour']
            else:
                self.segmented_image = None
                self.contour_image = None
            # 更新页数显示
            self.page_label.setText(f"{self.current_index + 1}/{len(self.image_files)}")    
            self.display_results()
            self.status_bar.showMessage(f"已加载: {os.path.basename(file_path)}")
            
        except Exception as e:
            self.status_bar.showMessage(f"加载错误: {str(e)}")

            
    def prev_image(self):
        """显示上一张图像"""
        if self.current_index > 0:
            self.current_index -= 1
            # 更新滑动条位置但不触发事件
            self.image_slider.blockSignals(True)
            self.image_slider.setValue(self.current_index)
            self.image_slider.blockSignals(False)
            self.load_current_image()
            
    def next_image(self):
        """显示下一张图像"""
        if self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            # 更新滑动条位置但不触发事件
            self.image_slider.blockSignals(True)
            self.image_slider.setValue(self.current_index)
            self.image_slider.blockSignals(False)
            self.load_current_image()
            
            # 如果是新图像且未处理，则自动处理
            if self.current_index >= len(self.batch_results):
                self.process_image(show_progress=True)  # 明确传递show_progress=True
                self.batch_results.append({
                    'segmented': self.segmented_image,
                    'contour': self.contour_image
                })
                self.progress_bar.setValue(int((len(self.batch_results) / len(self.image_files)) * 100))

    def slider_changed(self, value):
        """滑动条值改变事件"""
        if self.image_files and 0 <= value < len(self.image_files):
            self.current_index = value
            self.load_current_image()

    def clear_images(self):
        """清除所有图像显示和状态"""
        # 清除图像显示
        self.original_label.clear()
        self.segmented_label.clear()
        self.contour_label.clear()
        
        # 重置图像数据
        self.original_image = None
        self.segmented_image = None 
        self.contour_image = None
        
        # 重置批量处理状态
        self.image_files = []
        self.batch_results = []
        self.current_index = 0
        
        # 重置UI状态
        self.progress_bar.setValue(0)
        self.image_slider.setRange(0, 0)
        self.page_label.setText("0/0")
        self.save_btn.setEnabled(False)
        
        # 更新状态栏
        self.status_bar.showMessage("已清除所有图像")

    def set_light_theme(self):
        """设置白天主题"""
        self.setStyleSheet("")
        self.is_dark_mode = False
        self.status_bar.showMessage("已切换至白天模式")

        # 强制更新所有部件布局
        self.main_widget.updateGeometry()
        self.original_scroll.updateGeometry()
        self.segmented_scroll.updateGeometry()
        self.contour_scroll.updateGeometry()
        
        # 强制处理所有待处理事件
        QApplication.processEvents()
        
        # 强制重新计算和显示当前图像
        if self.original_image is not None:
            # 强制重置标签大小
            self.original_label.setMinimumSize(1, 1)
            self.segmented_label.setMinimumSize(1, 1)
            self.contour_label.setMinimumSize(1, 1)
            
            # 强制更新标签几何属性
            self.original_label.updateGeometry()
            self.segmented_label.updateGeometry()
            self.contour_label.updateGeometry()
            
            # 强制重新显示
            self.display_image(self.current_image)
        
        self.setStyleSheet("")
        self.is_dark_mode = False
        self.status_bar.showMessage("已切换至白天模式")

    def set_dark_theme(self):
        """设置夜间主题"""
        dark_style = """
        QMainWindow {
            background-color: #2D2D2D;
        }
        QMenuBar {
            background-color: #3D3D3D;
            color: #E0E0E0;
            border-bottom: 1px solid #555;
        }
        QMenuBar::item {
            background-color: transparent;
            padding: 5px 10px;
        }
        QMenuBar::item:selected {
            background-color: #505050;
        }
        QMenuBar::item:pressed {
            background-color: #606060;
        }
        QMenu {
            background-color: #3D3D3D;
            border: 1px solid #555;
            color: #E0E0E0;
        }
        QMenu::item:selected {
            background-color: #505050;
        }
        QComboBox {
            background-color: #505050;
            border: 1px solid #555;
            color: #E0E0E0;
            padding: 1px 1px 1px 3px;
        }
        QComboBox::drop-down {
            border: 0px;
        }
        QComboBox::down-arrow {
            image: url(images/down_arrow_white.png);
            width: 12px;
            height: 12px;
        }
        QComboBox QAbstractItemView {
            background-color: #505050;
            border: 1px solid #555;
            color: #E0E0E0;
            selection-background-color: #606060;
            outline: none;
        }
        QProgressBar {
            border: 1px solid #555;
            border-radius: 3px;
            text-align: center;
            color: #E0E0E0;
            background-color: #3D3D3D;
        }
        QProgressBar::chunk {
            background-color: #4CAF50;
            border-radius: 2px;
            border: 1px solid #3D3D3D;
        }
        QTitleBar {
            background-color: #3D3D3D;
            color: #E0E0E0;
        }
        QLabel, QPushButton, QComboBox, QSlider, QGroupBox {
            color: #E0E0E0;
        }
        QGroupBox {
            background-color: #3D3D3D;
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 3px;
        }
        QPushButton {
            background-color: #505050;
            border: 1px solid #555;
            padding: 5px;
        }
        QPushButton:hover {
            background-color: #606060;
        }
        QComboBox {
            background-color: #505050;
            border: 1px solid #555;
        }
        QScrollArea {
            background-color: #252525;
        }
        QLabel[objectName^="original_label"],
        QLabel[objectName^="segmented_label"],
        QLabel[objectName^="contour_label"] {
            background-color: #252525;
        }
        QSlider::groove:horizontal {
            background: #505050;
            height: 6px;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #4CAF50;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #FFFFFF;
            width: 16px;
            height: 16px;
            margin: -5px 0;
            border-radius: 8px;
        }
        QStatusBar {
            background-color: #3D3D3D;
            color: #E0E0E0;
        }
        """
        

        # 强制更新所有部件布局
        self.main_widget.updateGeometry()
        self.original_scroll.updateGeometry()
        self.segmented_scroll.updateGeometry()
        self.contour_scroll.updateGeometry()
        
        # 强制处理所有待处理事件
        QApplication.processEvents()
        
        # 强制重新计算和显示当前图像
        if self.original_image is not None:
            # 强制重置标签大小
            self.original_label.setMinimumSize(1, 1)
            self.segmented_label.setMinimumSize(1, 1)
            self.contour_label.setMinimumSize(1, 1)
            
            # 强制更新标签几何属性
            self.original_label.updateGeometry()
            self.segmented_label.updateGeometry()
            self.contour_label.updateGeometry()
            
            # 强制重新显示
            self.display_image(self.current_image)
        
        self.setStyleSheet(dark_style)
        self.is_dark_mode = True
        self.status_bar.showMessage("已切换至夜间模式")
            
        
        

    def show_about(self):
        """显示关于对话框"""
        about_text = """
        <b>胃镜图像息肉分割系统</b><br><br>
        版本: 2.0<br>
        作者: 任威名<br>
        邮箱: renweiming@tju.edu.cn<br>
        单位: 天津大学生物医学工程系<br>
        日期: 2025年5月<br><br>
        免责声明：本系统结果仅供临床参考<br>
        """
        # 创建消息框
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("关于")
        msg_box.setText(about_text)
        
        # 根据当前主题设置样式
        if self.is_dark_mode:
            msg_box.setStyleSheet("""
                QMessageBox {
                    background-color: #505050;
                }
                QLabel {
                    color: #E0E0E0;
                }
                QPushButton {
                    background-color: #2D2D2D;
                    border: 1px solid #555;
                    color: #E0E0E0;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #606060;
                }
            """)
        
        msg_box.exec_()        
            
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PolypSegmentationUI()
    window.show()
    sys.exit(app.exec_())