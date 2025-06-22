from graphviz import Digraph

# 创建图对象，垂直布局 + 高字体分辨率
dot = Digraph(comment='SE-ResNet50-UNet-Adjusted', format='svg')  # 修改为 SVG
# 调整 nodesep 和 ranksep
dot.attr(rankdir='TB', nodesep='1.2', ranksep='0.2')
#dot.attr(dpi='300')  # 对 SVG 可保留该设置或删除，SVG 可缩放
dot.attr('node', shape='box', fontname='Times-Roman', fontsize='14',
         height='0.4', width='0.6', style='filled', color='black', fillcolor='lightgray')

# 编码器部分
dot.node('Input', 'Input\n[1, H, W]', fillcolor='lightyellow')
dot.node('E0', 'Conv7x7\n[64, H/2, W/2]', fillcolor='#FFD699')
dot.node('E1', 'ResBlock1\n[256, H/4, W/4]', fillcolor='#FFD699')
dot.node('SE1', 'SE', fillcolor='#90EE90')
dot.node('E2', 'ResBlock2\n[512, H/8, W/8]', fillcolor='#FFD699')
dot.node('SE2', 'SE', fillcolor='#90EE90')
dot.node('E3', 'ResBlock3\n[1024, H/16, W/16]', fillcolor='#FFD699')
dot.node('SE3', 'SE', fillcolor='#90EE90')
dot.node('E4', 'ResBlock4\n[2048, H/32, W/32]', fillcolor='#FFD699')
dot.node('SE4', 'SE', fillcolor='#90EE90')

# 解码器部分
dot.node('D1', 'UpConv1\n[1024, H/16, W/16]')
dot.node('DSE1', 'SE', fillcolor='#90EE90')
dot.node('D2', 'UpConv2\n[512, H/8, W/8]')
dot.node('DSE2', 'SE', fillcolor='#90EE90')
dot.node('D3', 'UpConv3\n[256, H/4, W/4]')
dot.node('DSE3', 'SE', fillcolor='#90EE90')
dot.node('D4', 'UpConv4\n[64, H/2, W/2]')
dot.node('DSE4', 'SE', fillcolor='#90EE90')

# 编码路径
dot.edge('Input', 'E0', label='Conv7x7')  
dot.edge('E0', 'E1', label='MaxPool+ResBlock1')
dot.edge('E1', 'SE1')
dot.edge('SE1', 'E2', label='ResBlock2')
dot.edge('E2', 'SE2')
dot.edge('SE2', 'E3', label='ResBlock3')
dot.edge('E3', 'SE3')
dot.edge('SE3', 'E4', label='ResBlock4')
dot.edge('E4', 'SE4')

# 解码路径
dot.edge('SE4', 'D1', label='Up+Concat SE3')
dot.edge('D1', 'DSE1')
dot.edge('DSE1', 'D2', label='Up+Concat SE2')
dot.edge('D2', 'DSE2')
dot.edge('DSE2', 'D3', label='Up+Concat SE1')
dot.edge('D3', 'DSE3')
dot.edge('DSE3', 'D4', label='Up+Concat E0')
dot.edge('D4', 'DSE4')

# 添加最终输出部分
dot.node('FinalUp', 'Upsample\n[64, H, W]')
dot.node('Output', 'Output\n[1, H, W]', fillcolor='lightyellow')
dot.edge('DSE4', 'FinalUp', label='Upsample x2', fontsize='12')
dot.edge('FinalUp', 'Output', label='1x1 Conv', fontsize='12')

# 跳跃连接
dot.edge('SE3', 'D1', style='dashed', color='gray', label='skip')
dot.edge('SE2', 'D2', style='dashed', color='gray', label='skip')
dot.edge('SE1', 'D3', style='dashed', color='gray', label='skip')
dot.edge('E0', 'D4', style='dashed', color='gray', label='skip')

# 输出为 SVG 格式图像
dot.render('se_resnet50_unet_adjusted_highres', format='svg', cleanup=False)
