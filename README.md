# Wasteer Task
This repository is created for the technical interview for Wasteer. The task description is as follows: 

In this README.md you will find the solutions steps that were taken while solving the problems in the given task. I wanted to avoid spagetti code, therefore, I tried to use OOP paradigm as much as possible. However, since the task requires a demo for the fps most of the solution consists of Python scripts. 

### Steps:

1. Research
2. Fine-Tuning
3. Performance Test
4. Improving FPS
5. Future Work

## 1. Research:
The initial step I take when starting an CV assignment is to understand the task by analyzing the dataset and research for the existing work. 

When I analyze the data.yaml file, I saw that there were only one class that is 'pallet'. The dataset was already seperated as train, test, val.   

When I start looking for wooded pallet detection, I found a pre-trained weights for yolov8 from Roboflow web-site: https://blog.roboflow.com/logistics-object-detection-model/ . When I test the pretrained model I saw that it was working fine but still need some fine-tuning. Sample results from the pre-trained model:

<p float="left">
    <img src="/images/sample_1.png" width="400">
    <img src="/images/sample_2.png" width="400">
</p>

As it can be seen from the sample images the model detected also other classes such as forklift and person. 


## 2. Fine-Tuning:
I used the pre-trained weighs and started a short training for 50 epocs. Following shows the training results:
![Training metrics.](/images/results.png "This image shows the training metrics.")

### There were three main reasons behind training the model again:
1.  **Having an end-to-end model for specific class:** The pre-trained model has 20 classes and wooden pallets was one of them. Therefore, the output may need more post-processing to obtain specific class. Also, the loss function may focus on other classes during backpropogating which means there is a small possibility of imroving the performance for only one class.
2. **Data augmentation**: The number of training data will be more in numbers because of adding data augmentation. ultralytics's mosaic augmentation known one of the most helpful one. So, I used it as well as the classical ones. 
3. **Fine-tuning**: The distribution of the Wood Pallet class differs significantly from the original dataset, therefore, fine-tuning was essential.

## 3. Performance Test:
The performance of the fine-tuned model and pre-trained model are as follows:
| model    | Precision | Recall | mAP |
| -------- |:---------:| :------:| :------:|
| pre-trained| right fo| right fo| right fo|
| fine-tuned | right ba| right fo| right fo|

## 4. Improving FPS:
FPS is very important when it comes to the real-time scenarios. One of the most popular way to improve FPS of a .pt model is to convert the model to .onnx and use onnxruntime for inferencing. Here the fps results for M2 CPU, V100GPU and finally onnxruntime. 

### M2 CPU
| model  | fps |
| ------ |:-------------:|
| .pt    | 14.351     |
| .onnx  | 26.189     |
| .half_onnx | 26.752     |

### V100 GPU
| model  | fps |
| ------ |:-------------:|
| .pt    | 41.290     |
| .onnx  | 42.469     |
| .half_onnx | 42.510     |

### onnxruntime on V100 GPU
| model  | fps |
| ------ |:-------------:|
| .onnx  | 151.273  |
| .half_onnx | 154.142 |

<p float="left">
    <img src="/images/onnx.png" width="400">
    <img src="/images/onnx_half.png" width="400">
</p>
