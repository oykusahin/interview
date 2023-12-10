import os
import time
from PIL import Image
from pathlib import Path
from ultralytics import YOLO

class YoloInference:
    def __init__(self, model_path, test_images_folder):
        self.model = YOLO(model_path)
        self.test_images_folder = test_images_folder

    def load_dummy_image(self):
        dummy_image_path = os.path.join(self.test_images_folder, os.listdir(self.test_images_folder)[0])
        dummy_image = Image.open(dummy_image_path)
        dummy_image = dummy_image.resize((320, 320))
        return dummy_image

    def load_and_infer_images(self):
        num_frames = len(os.listdir(self.test_images_folder))
        start_time = time.time()

        for image_name in os.listdir(self.test_images_folder):
            image_path = os.path.join(self.test_images_folder, image_name)
            image = Image.open(image_path)
            image = image.resize((320, 320))
            
            # Perform inference
            results = self.model(image)

        end_time = time.time()
        elapsed_time = end_time - start_time

        fps = num_frames / elapsed_time
        return fps