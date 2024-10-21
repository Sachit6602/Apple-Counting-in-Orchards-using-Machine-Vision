import torch
from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.plots import plot_one_box
import cv2
import os
import json
import csv
from sklearn.metrics import precision_score, recall_score, f1_score

weights = 'C:/Users/Lenovo/Downloads/MV/yolov7s.pt'
source_dir = 'C:/Users/Lenovo/Downloads/MV'
output_dir = 'C:/Users/Lenovo/Downloads/MV/output'
json_annotation_file = 'C:/Users/Lenovo/Downloads/MV/ground_truth_annotations.json'
csv_annotation_file = 'C:/Users/Lenovo/Downloads/MV/annotations.csv'

with open(json_annotation_file, 'r') as json_file:
    data = json.load(json_file)

images = data.get("images", [])
annotations = data.get("annotations", [])

image_id_to_filename = {image["id"]: image["filename"] for image in images}

image_id_to_bboxes = {}
for annotation in annotations:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    if image_id not in image_id_to_bboxes:
        image_id_to_bboxes[image_id] = []
    image_id_to_bboxes[image_id].append(bbox)

with open(csv_annotation_file, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['image_id', 'class', 'x', 'y', 'width', 'height'])

    for image_id, bboxes in image_id_to_bboxes.items():
        filename = image_id_to_filename.get(image_id, "")

        for bbox in bboxes:
            x, y, width, height = bbox
            csv_writer.writerow([filename, 'apple', x, y, width, height])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = attempt_load(weights, map_location=device)
model.eval()

os.makedirs(output_dir, exist_ok=True)

all_predictions = []
all_annotations = []

img_formats = ['png', 'jpg', 'jpeg']
img_files = [f for f in os.listdir(source_dir) if f.split('.')[-1].lower() in img_formats]

for img_file in img_files:
    img_path = os.path.join(source_dir, img_file)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)[0]

    pred_boxes = pred[:, :4].cpu().numpy()
    pred_scores = pred[:, 4].cpu().numpy()

    img_name = os.path.splitext(img_file)[0]

    gt_boxes = []
    for annotation in annotations:
        if annotation['image_id'] == img_name:
            gt_boxes.append(annotation['bbox'])

    all_predictions.extend(pred_boxes)
    all_annotations.extend(gt_boxes)

    img = cv2.imread(img_path)
    for det in pred:
        det = det.cpu().numpy()
        x1, y1, x2, y2, conf, cls = det
        plot_one_box([x1, y1, x2, y2], img, label=f'Class {int(cls)}', color=(0, 255, 0))

    save_path = os.path.join(output_dir, f'{img_name}_output.jpg')
    cv2.imwrite(save_path, img)

y_true = torch.tensor([1] * len(all_annotations), device=device, dtype=torch.float32)
y_pred = torch.tensor([1] * len(all_predictions), device=device, dtype=torch.float32)

precision = precision_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
recall = recall_score(y_true.cpu().numpy(), y_pred.cpu().numpy())
f1 = f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy())

