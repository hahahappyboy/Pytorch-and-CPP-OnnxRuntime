import torch
import torch.nn as nn
import functools

from Pix2pix_GlobalGenerator import GlobalGenerator

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':# yes
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# 创建模型，这些都是从官方源码上复制粘贴过来的
norm_layer = get_norm_layer(norm_type='instance')
netG = GlobalGenerator(3, 3, 64, 4, 9, norm_layer)
# 加载参数
save_path = 'pix2pixHDCartoon.pth'
netG.load_state_dict(torch.load(save_path))
# 导出onnx模型
input_names = ['inputs'] # 输入名字
output_names = ['outputs'] # 输出名字
# batchsize多少决定你c++使用onnx模型时一次处理多少图片
batch_size = 1
# 输入图片的通道 高 宽
c, h, w= 3, 256, 256
dummy_input = torch.randn(batch_size, c, h, w, requires_grad=True)
torch.onnx.export(netG, dummy_input, "pix2pixHD_Cartoon_batch1.onnx", verbose=True,
                  input_names=input_names,output_names=output_names)