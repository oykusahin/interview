import os

from ultralytics import YOLO
from modify_folder import process_files_in_folder

# Path for the project directory
path = '/Users/oyku/Documents/Projects/interview/'

"""
 Load the YOLO model from Roboflow. 
 The model was trained for Logistics and includes wooded pallet class.
 Therefore, I tought I can be good point to start.
"""
model = YOLO(path + 'roboflow_best.pt')

"""
 Predict the test dataset on model without any fine-tuning or adjustment.
 Since the model was taken from Roboflow, I thought the initial step is to test its perfomance on wooden pallets. 
 So, I limit the test for only the wooden pallet which has class_id: 19
"""
prediction_path = '/runs/detect/initial_results/labels'
if not os.path.exists(prediction_path):
    source = path+'pallets/test/images/'
    model.predict(source, save=True, imgsz=640, batch=16, conf=0.5, plots=True, classes=[19], save_txt=True)

"""
 Since YOLOv8 is not able to provide results before fine-tuning, I added a small script to see the results. 
 Let me show an example from the results of the predictions from the previous step:
    19 0.588137 0.399435 0.378522 0.747793
 And the results from actual dataset were: 
    0 0.5859375 0.40078125 0.371875 0.75625
 As can be seen, if we only change the class name then we can calculate the perfomance of the model.
 The following script changes the class names from 19 to 0 
"""
if not os.path.exists(prediction_path):
    folder_path = path + prediction_path
    process_files_in_folder(folder_path)

"""
 Now, the predicted dataset can be easily used to measure the performance of the Roboflow model.
"""

