from yolov8.inference import onnx_inference as oinf
from yolov8.inference import pt_inference as pinf
from ultralytics import YOLO

import os

PATH = os.getcwd()

def count_files_in_folder(folder_path):
    try:
        file_count = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        return file_count
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def calculate_accuracy():
    print("Calculating accuracy...")
    
    model_path = PATH + '/model/half_long_fine_tuned.onnx'
    test_images = PATH + '/wooden_pallets.yaml'
    
    # Load a model
    model = YOLO(model_path)  # load a custom model
    
    # Validate the model
    metrics = model.val(data=test_images) 
    metrics.box.map    # map50-95
    metrics.box.map50  # map50
    metrics.box.map75  # map75
    metrics.box.maps   # a list contains map50-95 of each category

def calculate_fps():
    print("Calculating frames per second...")
    
    model_path = PATH + '/model/half_long_fine_tuned.onnx'
    test_images = PATH + '/dataset/test/images/'
    num_batches = count_files_in_folder(test_images)

    onnx_inference = oinf.OnnxInference(model_path, test_images, num_batches)
    fps_result = onnx_inference.run_inference()
    print(f"FPS: {fps_result}")

def main():
    
    while True:
        user_choice = input("Enter 'acc' for accuracy, 'fps' for frames per second, or 'exit' to quit: ").lower()

        if user_choice == 'acc':
            calculate_accuracy()
        elif user_choice == 'fps':
            calculate_fps()
        elif user_choice == 'exit':
            print("Exiting the demo.")
            break
        else:
            print("Invalid choice. Please enter 'acc', 'fps', or 'exit'.")

if __name__ == "__main__":
    main()
