import onnxruntime
import numpy as np
from PIL import Image
import time

path = '/content/drive/MyDrive/YOLOv8'
onnx_model_path = path + '/runs/detect/train5/weights/best.onnx'

session_options = onnxruntime.SessionOptions()
session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session = onnxruntime.InferenceSession(onnx_model_path, session_options=session_options)

test_images_folder = test_images_folder = path + '/test/images/'

dummy_images = []
for i in range(1):
    dummy_image = np.random.rand(320, 320, 3).astype(np.float32)  # Replace with actual image loading
    dummy_images.append(dummy_image)

# Convert the list of images to a 4D NumPy array
input_images = np.array(dummy_images)

# ONNX Runtime expects input shape (batch_size, channels, height, width)
input_images = np.transpose(input_images, (0, 3, 1, 2))

# Move the model to GPU (if not already)
session.set_providers(['CUDAExecutionProvider'])

# Perform inference and measure FPS
input_name = session.get_inputs()[0].name

num_batches = 763 
start_time = time.time()

for _ in range(num_batches):
    outputs = session.run([], {input_name: input_images})

end_time = time.time()
elapsed_time = end_time - start_time

fps = num_batches / elapsed_time
print(f"FPS: {fps}")