from ultralytics import YOLO

# Path for the project directory
path = '/Users/oyku/Documents/Projects/interview/'

# Load the YOLO model from Roboflow. 
# The model was trained for Logistics and includes wooded pallet class.
# Therefore, I tought I can be good point to start.
model = YOLO(path + 'best.pt')

# Predict the test dataset on model without any fine-tuning or adjustment.
# Since the model was taken from Roboflow, I thought the initial step is to test its perfomance on wooden pallets. 
# So, I limit the test for only the wooden pallet which has class_id: 19
source = path+'pallets/test/images/'
model.predict(source, save=True, imgsz=640, batch=16, conf=0.5, plots=True, classes=[19], save_txt=True)