from ultralytics import YOLO

# Path for the project directory
path = '/Users/oyku/Documents/Projects/interview/'

# Load the YOLO model from Roboflow. 
# The model was trained for Logistics and includes wooded pallet class.
# Therefore, I tought I can be good point to start.
model = YOLO(path + 'roboflow_best.pt')

"""
 Data augmentation: please check the data augmentation hyperparameters from the wooden_pallets.yaml file
 The best training result will be used as fine-tuned model for the rest of the task. 
"""
model.train(data='wooden_pallets.yaml', epochs=50, device='mps', imgsz=640, batch=32, plots=True)
model.export(format='onnx', half=True, dynamic=False, imgsz=[320,320])

"""
 It is also important to validate the fine-tuned model.
 The following code is responsible for the validation. 
 As can be seen from the results both data-augmentation and fine-tuning improved the accuracy of the model. 
"""
valid_results = model.val()