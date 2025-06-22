import PyInstaller.__main__
import os
import shutil
import sys
import tempfile

# 项目根目录
project_root = os.path.dirname(os.path.abspath(__file__))

# 打包配置
PyInstaller.__main__.run([
    os.path.join(project_root, 'ui', 'polyp_segmentation_ui.py'),
    '--onefile',
    '--windowed',
    '--icon=' + os.path.join(project_root, 'ui', 'icon.ico'),
    '--add-data=' + os.path.join(project_root, 'best_model(43+val+de+AdamW).pth') + os.pathsep + '.',
    '--add-data=' + os.path.join(project_root, 'best_model_R_CBAM(44+val+de+AdamW).pth') + os.pathsep + '.',
    '--add-data=' + os.path.join(project_root, 'best_model_R_ECA(33+val+de+AdamW).pth') + os.pathsep + '.',
    '--add-data=' + os.path.join(project_root, 'best_model_R_SE(44+val+de+AdamW).pth') + os.pathsep + '.',
    '--add-data=' + os.path.join(project_root, 'best_model_ResNet50(34+val+de+Adam).pth') + os.pathsep + '.',
    '--add-data=' + os.path.join(project_root, 'ui', 'resource_loader.py') + os.pathsep + '.',
    '--name=PolypSegmentation',
    '--distpath=' + os.path.join(project_root, 'dist'),
    '--workpath=' + os.path.join(project_root, 'build'),
    '--hidden-import=torch',
    '--hidden-import=cv2',
    '--hidden-import=numpy',
    '--hidden-import=PyQt5.sip',
    '--hidden-import=psutil',
    '--hidden-import=torchvision',
    '--hidden-import=skimage',
    '--paths=' + os.path.join(project_root),
    '--clean',
    '--noconfirm'
])

# 创建资源解压辅助脚本
resource_loader_path = os.path.join(project_root, 'ui', 'resource_loader.py')
with open(resource_loader_path, 'w') as f:
    f.write('''import os
import sys
import tempfile

def get_resource_path(relative_path):
    """获取资源文件的绝对路径"""
    try:
        # PyInstaller创建的临时文件夹路径
        base_path = sys._MEIPASS
    except AttributeError:
        # 正常Python环境下的路径
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)
''')

print("打包完成！exe文件位于:", os.path.join(project_root, 'dist'))