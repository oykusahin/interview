# Wasteer Technical Task
This repository is created for Wasteer's technical interview. The task description is as follows: 

- Develop and demonstrate a real-time pallet detection system using the provided dataset.
- Balance the model’s speed and accuracy, with an emphasis on achieving high fps.
- Document your approach and any challenges faced.

In this README.md you will find the solutions steps that were taken while solving the given problems in the task and documentation of how to run it yourself. I wanted to avoid spagetti code, therefore, I tried to use OOP paradigm as much as possible. However, since the task requires a demo for measuring fps most of the solution consists of Python scripts. 

# Documentation
This documentation is prepared for Google Colab. 

1. Please change your runtime to one of the GPUs. 

2. Load the Drive helper and mount
```
from google.colab import drive
drive.mount('/content/drive')
```

3. Clone repository and install the requirements.txt
```
%cd drive/MyDrive/
!git clone https://github.com/oykusahin/interview  # clone
%cd interview
%pip install -r requirements.txt  # install
```

5. Check if the requirements were successfully installed.
```
import ultralytics
ultralytics.checks()
```

7. Install the dataset.
```
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(COMPANY_NAME).project(PROJECT_NAME)
dataset = project.version(2).download("TYPE_NAME")
```

6. Make sure you are using GPU.
```
!nvidia-smi
```

7. Run the code for simple demo.
```
!python main.py
```

### Steps:

1. Research
2. Fine-Tuning
3. Performance Test
4. Improving FPS
5. Future Work

## 1. Research:
The initial step I take when starting a computer vision assignment is to understand the task by analyzing the dataset and research for the existing work. 

When I analyzed the data.yaml file which came with the datasets itself, I saw that there were only one class and that was 'pallet'. The dataset was already seperated to train, test, valid groups.   

YOLOv8 is recommended for computer vision projects due to its high accuracy, with the YOLOv8m model achieving a 50.2% mAP on COCO and outperforming YOLOv5 on the Roboflow 100 dataset. Its developer-friendly features, including a user-friendly CLI and a well-structured Python package, contribute to a seamless coding experience, and the growing community around YOLO ensures ample guidance and support from experts in computer vision.

When I start looking for wooded pallet detection, I found a pre-trained weights for YOLOv8 from Roboflow web-site: https://blog.roboflow.com/logistics-object-detection-model/ . When I tested the pre-trained model I saw that it was working fine but still need some fine-tuning. Sample results from the pre-trained model:

<p float="left">
    <img src="/images/sample_1.png" width="400">
    <img src="/images/sample_2.png" width="400">
</p>

As it can be seen from the sample images, the model detected not only wooden pallet but also other classes such as forklift and person. 


## 2. Fine-Tuning:
I used the pre-trained weighs and started a short training for 50 epochs. Following shows the training results:
![Training metrics.](/images/results.png "This image shows the training metrics.")

### There were three main reasons behind training the model again:
1.  **Having an end-to-end model for specific class:** The pre-trained model had 20 classes and wooden pallets was one of them. Therefore, the output may need more post-processing to obtain that specific class. Also, the loss function may focused on other classes during backpropogation which means there was a small possibility of improving the performance for the pallet class.
2. **Data augmentation**: The number of training data will be more in numbers because of adding data augmentation. Ultralytics's mosaic augmentation known one of the most helpful one. So, I used it as well as the classical ones. 
3. **Fine-tuning**: The distribution of the Wood Pallet class differs significantly from the original dataset, therefore, fine-tuning was essential.

## 3. Performance Test:
The performance of the pre-trained and fine-tuned model are as follows:
| model    | Precision | Recall | F1 Score |
| -------- |:---------:| :------:| :------:|
| pre-trained| 0.99 | 1.00| right 0.99|
| fine-tuned | right ba| right fo| right fo|

## 4. Improving FPS:
FPS is very important when it comes to the real-life scenarios. One of the most popular way to improve fps of a .pt model is to convert the model to .onnx and use onnxruntime for inferencing. Here, the fps results for M2 CPU, V100GPU and finally onnxruntime. 

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

## 5. Future Work:
The following can be done as a future work for fps improvement while maintaining accuracy: 
1. Implementing NVIDIA DALI for pre-processing can enhance FPS by offloading data augmentation tasks to the GPU, reducing CPU overhead.
2. Exploring alternative inferencing technologies, such as TensorRT or OpenVINO, may optimize model execution and improve overall inference speed.
3. Incorporating angle-aware object detection techniques can enhance accuracy in detecting pallets, potentially improving model performance.
4. Implementing dynamic batching during inference can optimize resource utilization, leading to improved throughput without sacrificing accuracy.
5. Evaluating smaller YOLO variants for object detection may result in faster inference speeds while maintaining acceptable accuracy levels.
6. Investigating CPU-specific optimizations, such as OpenBLAS or MKL, can potentially improve inference speed on CPU-based systems.
