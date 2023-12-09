import os
import time
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

path = '/Users/oyku/Documents/Projects/interview/'
model = YOLO(path + '/models/fine_tuned.pt')
test_images_folder = path + '/pallets/test/images/'

dummy_image = Image.open(os.path.join(test_images_folder, os.listdir(test_images_folder)[0]))
dummy_image = dummy_image.resize((320, 320))
print((dummy_image.size))
model(dummy_image)

# Measure FPS
num_frames = len(os.listdir(test_images_folder))
start_time = time.time()

for image_name in os.listdir(test_images_folder):
    image_path = os.path.join(test_images_folder, image_name)
    image = Image.open(image_path)
    image = image.resize((320, 320))
    
    # Perform inference
    results = model(image)

end_time = time.time()
elapsed_time = end_time - start_time

fps = num_frames / elapsed_time
print(f"FPS: {fps}")