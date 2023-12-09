# Wasteer Task
This repository is created for the technical interview for Wasteer. The task description is as follows: 

In this README.md you will find the solutions steps that were taken while solving the problems in the given task. 

### Steps:

1. Research
2. Fine-Tuning
3. Performance Test
4. Improving FPS

## 1. Research:
The initial step I take when starting an CV assignment is to understand the task by analyzing the dataset and research for the existing work. 

When I analyze the data.yaml file, I saw that there were only one class that is 'pallet'. The dataset was already seperated as train, test, val.   

When I start looking for wooded pallet detection, I found a pre-trained weights for yolov8 from Roboflow web-site: https://blog.roboflow.com/logistics-object-detection-model/ . When I test the pretrained model I saw that it was working fine but still need some fine-tuning. Sample results from the pre-trained model:

![Sample 1.](/images/sample_1.png =100x20)
![Sample 2.](/images/sample_2.png =100x20)

As it can be seen from the sample images the model detected also other classes such as forklift and person. 


## 2. Fine-Tuning:
I used the pre-trained weighs and started a short training for 50 epocs. Following shows the training results:
![Training metrics.](/images/results.png "This image shows the training metrics.")

### There were three main reasons behind training the model again:
1. Having an end-to-end model for specific class: The pre-trained model has 20 classes and wooden pallets was one of them. Therefore, the output may need more post-processing to obtain specific class. Also, the loss function may focus on other classes during backpropogating which means there is a small possibility of imroving the performance for only one class.
2. Data Augmentation: The number of training data will be more in numbers because of adding data augmentation. ultralytics's mosaic augmentation known one of the most helpful one. So, I used it as well as the classical ones. 
3. Fine-tuning: the distribution of the Wood Pallet class differs significantly from the original dataset, therefore, fine-tuning was essential.

## 3. Performance Test:
