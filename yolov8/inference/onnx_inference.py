import onnxruntime
import numpy as np
import time

class OnnxInference:
    def __init__(self, onnx_model_path, test_images_folder, num_batches):
        self.session_options = onnxruntime.SessionOptions()
        self.session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = onnxruntime.InferenceSession(onnx_model_path, session_options=self.session_options)

        self.test_images_folder = test_images_folder
        self.num_batches = num_batches

        # Initialize dummy images (replace with actual loading logic)
        self.dummy_images = [np.random.rand(320, 320, 3).astype(np.float32) for _ in range(1)]

        # Convert the list of images to a 4D NumPy array
        self.input_images = np.array(self.dummy_images)

        # ONNX Runtime expects input shape (batch_size, channels, height, width)
        self.input_images = np.transpose(self.input_images, (0, 3, 1, 2))

        # Move the model to GPU (if not already)
        self.session.set_providers(['CUDAExecutionProvider'])

    def run_inference(self):
        input_name = self.session.get_inputs()[0].name

        start_time = time.time()

        for _ in range(self.num_batches):
            outputs = self.session.run([], {input_name: self.input_images})

        end_time = time.time()
        elapsed_time = end_time - start_time

        fps = self.num_batches / elapsed_time
        return fps