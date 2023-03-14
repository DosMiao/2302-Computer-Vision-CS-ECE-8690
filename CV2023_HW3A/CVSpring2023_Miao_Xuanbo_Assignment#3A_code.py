import cv2
import torchvision
import numpy as np
import os
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)

model.eval()

threshold = 0.8

cap = cv2.VideoCapture('PETS09-S2L1-raw.webm')

fourcc = cv2.VideoWriter_fourcc(*'VP90')
out = cv2.VideoWriter('output.webm',fourcc, 30.0, (640,480))
frame_id=0

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

        for box, label, score in zip(boxes, labels, scores):
            if label == 1 and score >= threshold: # 1 corresponds to the pedestrian class
                x1, y1, x2, y2 = box.astype(np.int32)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('frame',frame)
        frame_id+=1
        if 0:   #save all frames    [1, 100, 200, 400]
            filename = f"./frames/frame_{frame_id}.png";   
            cv2.imwrite(filename, frame)
        print(f"./frames/frame_{frame_id}.png")

        out.write(frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release the resources
cap.release()
out.release()
cv2.destroyAllWindows()
