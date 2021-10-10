# Pytorch-and-CPP-OnnxRuntime
@[TOC](Pytorch和C++OnnxRuntime使用方法)

## 写在前面
最近有个需求要将Pytorch训练好的模型迁移到C++上去使用，再网上查了一些资料结合上自己的实际情况最终使用onnxruntime迁移成功。借此机会给大家分享一下，**若有写的不对的地方请大家批评指正**！下面给大家看看运行结果

这是Pytorch的onnxruntime运行时间

![在这里插入图片描述](https://img-blog.csdnimg.cn/fe1275746d6c43e99543336c167eb6bb.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_13,color_FFFFFF,t_70,g_se,x_16)

这是C++的onnxruntime运行时间，感觉两者速度是差不多的~可能要大批量数据的时候C++才能体现出它的优势吧。

![在这里插入图片描述](https://img-blog.csdnimg.cn/103bb1ca1d2b47a09f4b7bbc7c2fe2e8.png)

这是onnx模型运行结果，左边是输入，右边是输出。这里Pytorch和C++的运行结果没有任何区别，因为是同一个onnx模型跑出来的。

![在这里插入图片描述](https://img-blog.csdnimg.cn/3d31cdad1e414fc891b03ee3a582efb0.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAaWlpaWlpaW1w,size_16,color_FFFFFF,t_70,g_se,x_16)

**具体细节请看：**
[https://blog.csdn.net/iiiiiiimp/article/details/120621682](https://blog.csdn.net/iiiiiiimp/article/details/120621682)

**训练好的模型以及导出的onnx模型在这：**
链接：[https://pan.baidu.com/s/1m35zq0wqTeaOZ5rj2L2dKA](https://pan.baidu.com/s/1m35zq0wqTeaOZ5rj2L2dKA) 
提取码：iimp 

**如何使用？**
改一下代码中的输入图片路径、模型路径、输出图片路径即可~

## 写在后面
目前我能想到的地方就这些了，如果错误的地方请多多包含啦~欢迎大家提问！



