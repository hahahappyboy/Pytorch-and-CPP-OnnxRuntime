import numpy
import onnxruntime
import torch
# 加载onnx模型
ort_session = onnxruntime.InferenceSession("pix2pixHD_Cartoon_batch1.onnx")

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# 验证模型
batch_size = 1
# 输入图片的通道 高 宽
c, h, w= 3, 256, 256
dummy_input = torch.randn(batch_size, c, h, w, requires_grad=True)
ort_inputs = {
    'inputs': to_numpy(dummy_input)
}
ort_outs = ort_session.run(None, ort_inputs)

print(numpy.array(ort_outs).shape)