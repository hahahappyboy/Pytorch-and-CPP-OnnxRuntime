import cv2
import onnxruntime

import torch
import numpy as np

import time

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


# 函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型。函数后面跟着的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型。
def normalize(tensor, mean, std, inplace=False):
    if not inplace:
        tensor = tensor.clone()
    dtype = tensor.dtype# torch.float64
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    tensor.sub_(mean).div_(std)
    return tensor

def to_tensor(pic):
    img = pic.transpose((2, 0, 1))
    img = torch.from_numpy(img) # uint8
    img = img.float().div(255) # float32
    return img

def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    image_numpy = image_tensor.astype(float)
    if normalize:
        #print(image_numpy.shape)
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)

    return image_numpy.astype(imtype)


print('预处理')
time_pre_start = time.time()
img = cv2.imread('A_01425.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

tensor_A = to_tensor(img)
label = normalize(tensor_A,(0.5, 0.5, 0.5),(0.5, 0.5, 0.5))

time_pre_end = time.time()
print('预处理时间:',time_pre_end-time_pre_start,'s')
print('预处理结束')
ort_session = onnxruntime.InferenceSession("pix2pixHD_Cartoon_batch1.onnx")
ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(label.unsqueeze(0)).astype(np.float32)}
print('cuda加速')
time_start=time.time()
ort_outs = ort_session.run(None, ort_inputs)
time_end=time.time()
print('加速时间',time_end-time_start,'s')
print('cuda加速结束')
print('保存结果')
time_back_start = time.time()
ort_outs = np.squeeze(ort_outs[0],axis=0)
img = tensor2im(ort_outs)
img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imwrite('A_01425_Cartoon.png',img)
time_back_end = time.time()
print('后处理时间:',time_back_end-time_back_start,'s')
print('运行结束')

