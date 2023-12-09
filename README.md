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

When I start looking for wooded pallet detection, I found a pre-trained weights for yolov8 from Roboflow web-site: https://blog.roboflow.com/logistics-object-detection-model/ . When I test the pretrained model I saw that it was working fine but still need some fine-tuning. 

## 2. Fine-Tuning:
I used the pre-trained weighs and started a short training for 50 epocs. Following shows the training results:
![Training metrics.](/images/results.png "This image shows the training metrics.")
