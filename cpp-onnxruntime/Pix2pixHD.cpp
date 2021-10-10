#include <iostream>
#include<opencv2/opencv.hpp>
#include<string>
#include<torch/torch.h>
#include<io.h>
#include <onnxruntime_cxx_api.h>
#include "cuda_provider_factory.h"

torch::Tensor to_tensor(cv::Mat img) {
	torch::Tensor img_tensor = torch::from_blob(img.data, { img.rows, img.cols, 3 }, torch::kByte);  // torch::kByte要对应cv::Mat img的数据类型
	img_tensor = img_tensor.permute({ 2, 0, 1 }); //转为C*H*W
	img_tensor = img_tensor.toType(torch::kFloat32);//转为
	img_tensor = img_tensor.div(255);
	return img_tensor;
}

torch::Tensor normalize(torch::Tensor tensor) {
	std::vector<double> mean = { 0.5,0.5,0.5 };
	std::vector<double> std = { 0.5,0.5,0.5 };
	tensor = torch::data::transforms::Normalize<>(mean, std)(tensor);// opencv
	return tensor;
}

int main()
{
	std::cout << "预处理" << std::endl;
    std::clock_t startTime_pre, endTime_pre;
    startTime_pre = clock();
	std::vector<cv::Mat> images = { //读取图片
		cv::imread("A_01425.png") ,
	};
	std::vector<float> inputVec;
	int batchSize = 1, channels, height, width;
	for (cv::Mat img : images) {
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		int w = img.cols;
		int h = img.rows;
		torch::Tensor tensor_A = to_tensor(img);//转为tensor，这样可以直接用归一化的api，方便对输入进行预处理
		torch::Tensor label = normalize(tensor_A);//归一化
		channels = label.sizes()[0], height = label.sizes()[1], width = label.sizes()[2];

		cv::Mat resultImg(h, w, CV_32FC3);
		/*std::memcpy 行中sizeof（）中内容，需要修改成c++中内建的数据类型，如果使用torch::kF32或者其他浮点型，会出现数据复制缺失的情况。*/
		std::memcpy((void*)resultImg.data, label.data_ptr(), sizeof(float) * label.numel());
		//构造模型输入std::vector<float> inputVec
		std::vector<cv::Mat> channels;
		cv::split(resultImg, channels);
		cv::Mat blue, green, red;
		blue = channels.at(0);
		green = channels.at(1);
		red = channels.at(2);
		std::vector<float> inputVec_red = (std::vector<float>)(blue.reshape(1, 1));
		std::vector<float> inputVec_green = (std::vector<float>)(green.reshape(1, 1));
		std::vector<float> inputVec_blue = (std::vector<float>)(red.reshape(1, 1));
		inputVec.insert(inputVec.end(), inputVec_red.begin(), inputVec_red.end());
		inputVec.insert(inputVec.end(), inputVec_green.begin(), inputVec_green.end());
		inputVec.insert(inputVec.end(), inputVec_blue.begin(), inputVec_blue.end());
	}

	endTime_pre = clock();
	std::cout << "预处理时间：" << (double)(endTime_pre - startTime_pre) / CLOCKS_PER_SEC << "s" << std::endl;
	std::cout << "预处理结束" << std::endl;

	std::vector<const char*> inputNames = { "inputs" };
	std::vector<const char*> outputNames = { "outputs" };
	OrtCUDAProviderOptions cuda_options{
	  0,
	  OrtCudnnConvAlgoSearch::EXHAUSTIVE,
	  std::numeric_limits<size_t>::max(),
	  0,
	  true
	};

	Ort::Env env = Ort::Env{ ORT_LOGGING_LEVEL_ERROR, "Default" };
	Ort::SessionOptions opt;
	opt.AppendExecutionProvider_CUDA(cuda_options);
	Ort::Session session(env, L"pix2pixHD_Cartoon_batch1.onnx", opt);

	std::vector<int64_t> inputSize = { batchSize, channels, height, width };
	rsize_t inputSizeCount = batchSize * channels * height * width;
	auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
	std::clock_t startTime, endTime;
	Ort::Value inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputVec.data(), inputSizeCount, inputSize.data(), inputSize.size());
	startTime = clock();
	std::cout << "cuda加速" << std::endl;
	std::vector<Ort::Value>  outputTensors = session.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &inputTensor, inputNames.size(), outputNames.data(), outputNames.size());
	endTime = clock();
	std::cout << "加速时间:" << (double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << std::endl;
	std::cout << "cuda加速结束" << std::endl;
	float* output = outputTensors[0].GetTensorMutableData<float>();

	std::vector<cv::Mat> results;
	std::cout << "保存结果" << std::endl;
	std::clock_t startTime_end, endTime_end;
	startTime_end = clock();
	int idx = 0;
	for (int b = 0; b < batchSize; b++) {

		torch::Tensor result_r = torch::from_blob(output, { height,width,1 });
		torch::Tensor result_g = torch::from_blob(&output[height * width - 1], { height,width,1 });
		torch::Tensor result_b = torch::from_blob(&output[height * width * 2 - 1], { height,width,1 });
		torch::Tensor result = torch::cat({ result_r, result_g, result_b }, 2);
		result = result.add(1).div(2).mul(255);
		result = result.clamp(0, 255);
		result = result.to(torch::kU8);
		cv::Mat resultImg(height, width, CV_8UC3);
		std::memcpy((void*)resultImg.data, result.data_ptr(), sizeof(torch::kU8) * result.numel());
		cv::cvtColor(resultImg, resultImg, cv::COLOR_RGB2BGR);
		results.push_back(resultImg);
	
	}
	cv::imwrite("A_01425_Cartoon.png", results[0]);
	endTime_end = clock();

	std::cout << "后处理时间：" << (double)(endTime_end - startTime_end) / CLOCKS_PER_SEC << "s" << std::endl;
	std::cout << "运行结束" << std::endl;


}