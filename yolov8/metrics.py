from sklearn.metrics import precision_recall_fscore_support
import os

def read_yolo_annotations(annotation_path):
    with open(annotation_path, 'r') as file:
        lines = file.readlines()
    annotations = [list(map(float, line.strip().split())) for line in lines if len(line.strip().split()) == 5]
    return annotations

def calculate_metrics(ground_truth_annotations, predicted_annotations, class_label):
    gt_boxes = [box for box in ground_truth_annotations if int(box[0]) == class_label]
    pred_boxes = [box for box in predicted_annotations if int(box[0]) == class_label]

    gt_boxes_count = len(gt_boxes)
    pred_boxes_count = len(pred_boxes)

    if gt_boxes_count == 0 and pred_boxes_count == 0:
        precision = recall = f1 = 1.0
    else:
        true_positives = sum([1 for pred_box in pred_boxes if any(iou(pred_box[1:], gt_box[1:]) >= 0.5 for gt_box in gt_boxes)])
        false_positives = pred_boxes_count - true_positives
        false_negatives = max(0, gt_boxes_count - true_positives)  # Adjusted this line

        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1

def iou(box1, box2):
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

def main():
    # Specify paths to annotation files
    ground_truth_path = '/Users/oyku/Documents/Projects/interview/pallets/test/labels'
    predicted_path = '/Users/oyku/Documents/Projects/interview/runs/detect/initial_results/labels'

    # Read annotations
    ground_truth_annotations = []
    for file_name in os.listdir(ground_truth_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(ground_truth_path, file_name)
            ground_truth_annotations.extend(read_yolo_annotations(file_path))

    predicted_annotations = []
    for file_name in os.listdir(predicted_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(predicted_path, file_name)
            predicted_annotations.extend(read_yolo_annotations(file_path))

    # Specify the class label for which you want to calculate metrics
    class_label = 0  # Replace with your desired class label

    # Calculate precision, recall, and F1 score
    precision, recall, f1 = calculate_metrics(ground_truth_annotations, predicted_annotations, class_label)

    print(f"Class {class_label} Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
