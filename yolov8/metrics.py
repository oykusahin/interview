from sklearn.metrics import precision_recall_fscore_support
import os

CLASS_LABEL = 0 

class YoloMetricsCalculator:
    def __init__(self, ground_truth_path, predicted_path):
        self.ground_truth_path = ground_truth_path
        self.predicted_path = predicted_path
        self.class_label = CLASS_LABEL

    def read_yolo_annotations(self, annotation_path):
        with open(annotation_path, 'r') as file:
            lines = file.readlines()
        annotations = [list(map(float, line.strip().split())) for line in lines if len(line.strip().split()) == 5]
        return annotations

    def calculate_metrics(self, ground_truth_annotations, predicted_annotations):
        gt_boxes = [box for box in ground_truth_annotations if int(box[0]) == self.class_label]
        pred_boxes = [box for box in predicted_annotations if int(box[0]) == self.class_label]

        gt_boxes_count = len(gt_boxes)
        pred_boxes_count = len(pred_boxes)

        if gt_boxes_count == 0 and pred_boxes_count == 0:
            precision = recall = f1 = 1.0
        else:
            true_positives = sum([1 for pred_box in pred_boxes if any(self.iou(pred_box[1:], gt_box[1:]) >= 0.5 for gt_box in gt_boxes)])
            false_positives = pred_boxes_count - true_positives
            false_negatives = max(0, gt_boxes_count - true_positives)

            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return precision, recall, f1

    def iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        x_left = max(x1 - w1 / 2, x2 - w2 / 2)
        y_top = max(y1 - h1 / 2, y2 - h2 / 2)
        x_right = min(x1 + w1 / 2, x2 + w2 / 2)
        y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

        intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def calculate_and_print_metrics(self):
        ground_truth_annotations = []
        for file_name in os.listdir(self.ground_truth_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.ground_truth_path, file_name)
                ground_truth_annotations.extend(self.read_yolo_annotations(file_path))

        predicted_annotations = []
        for file_name in os.listdir(self.predicted_path):
            if file_name.endswith('.txt'):
                file_path = os.path.join(self.predicted_path, file_name)
                predicted_annotations.extend(self.read_yolo_annotations(file_path))

        precision, recall, f1 = self.calculate_metrics(ground_truth_annotations, predicted_annotations)

        print(f"Class {self.class_label} Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")