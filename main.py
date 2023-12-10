from yolov8.inference import onnx_inference as oinf
from yolov8.inference import pt_inference as pinf
from yolov8 import metrics

PATH = '/content/drive/MyDrive/YOLOv8'

def calculate_accuracy():
    print("Calculating accuracy...")
    
    model_path = PATH + '/models/half_long_fine_tuned.onnx'
    test_images_folder = PATH + '/pallets/test/images/'

    yolo_inference = pinf.YoloInference(model_path, test_images_folder)
    dummy_image = yolo_inference.load_dummy_image()
    yolo_results = yolo_inference.load_and_infer_images()

    print(f"FPS: {yolo_results}")

    ground_truth_path = PATH + '/pallets/test/labels'
    predicted_path = PATH + '/runs/detect/detect/labels'
    class_label = 0

    metrics_calculator = metrics.YoloMetricsCalculator(ground_truth_path, predicted_path)
    metrics_calculator.calculate_and_print_metrics()


def calculate_fps():
    print("Calculating frames per second...")
    
    model_path = PATH + '/models/best.onnx'
    test_images = PATH + '/pallets/test/images/'
    num_batches = 763

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
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please enter 'acc', 'fps', or 'exit'.")

if __name__ == "__main__":
    main()
