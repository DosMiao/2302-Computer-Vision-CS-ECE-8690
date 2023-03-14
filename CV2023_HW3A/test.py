import cv2
import torchvision
import numpy as np
import os
import torch
from itertools import combinations
from scipy.spatial import distance

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

model.eval()

threshold = 0.8

cap = cv2.VideoCapture('PETS09-S2L1-raw.webm')

fourcc = cv2.VideoWriter_fourcc(*'VP90')
out = cv2.VideoWriter('output.webm',fourcc, 30.0, (640,480))
frame_id=0

# Define the Centroid Tracker object
class CentroidTracker:
    def __init__(self, max_disappeared=10):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            input_centroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i in range(len(object_centroids)):
                for j in range(len(input_centroids)):
                    D[i, j] = np.sqrt(np.sum((object_centroids[i] - input_centroids[j]) ** 2))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0

                used_rows.add(row)
                used_cols.add(col)

            unused_rows = set(range(len(object_centroids))) - used_rows
            unused_cols = set(range(len(input_centroids))) - used_cols

            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)

            for col in unused_cols:
                self.register

ct = CentroidTracker()
prev_objects = {}

while cap.isOpened():

    ret, frame = cap.read()

    if ret:
        img_tensor = torchvision.transforms.functional.to_tensor(frame)

        img_tensor = img_tensor.to(device)

        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pred = model(img_tensor)

        boxes = pred[0]['boxes'].detach().cpu().numpy()
        labels = pred[0]['labels'].detach().cpu().numpy()
        scores = pred[0]['scores'].detach().cpu().numpy()

        rects = []
        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score >= threshold: # 1 corresponds to the pedestrian class
                x1, y1, x2, y2 = box.astype(np.int32)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                rects.append((x1, y1, x2, y2))

        objects = ct.update(rects)

        if objects is not None:
            for object_id, centroid in objects.items():
                cv2.putText(frame, f'ID {object_id}', (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if object_id in prev_objects:
                    prev_centroid = prev_objects[object_id]
                    distance_pixels = distance.euclidean(centroid, prev_centroid)
                    speed_pixels_per_frame = distance_pixels / 1.0   # assuming 1 frame per second
                    speed_meters_per_second = speed_pixels_per_frame * pixels_to_meters_ratio
                    cv2.putText(frame, f'Speed {speed_meters_per_second:.2f} m/s', (centroid[0] - 10, centroid[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                prev_objects[object_id] = centroid

        cv2.imshow('frame',frame)
        if 0:   #save all frames    [1, 100, 200, 400]
            filename = f"./frames/frame_{frame_id}.png"
            cv2.imwrite(filename, frame)
            frame_id+=1

        out.write(frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release
cap.release()
out.release()
cv2.destroyAllWindows()