import random
from typing import Tuple

import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO


def gen_rgb() -> Tuple[int, int, int]:
    return (
        random.choice(range(256)),
        random.choice(range(256)),
        random.choice(range(256)),
    )


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Uncomment to use image or video
# cap = cv2.VideoCapture("YOUR PATH")

# List Yolo models
# https://docs.ultralytics.com/models/yolov8/#supported-tasks-and-modes
model = YOLO("yolov8n.pt")
min_confidence = 0.4

class_names = model.names
class_colors = [gen_rgb() for _ in class_names]
print("Model class names:")
print(list(class_names.values()))

while True:
    success, image = cap.read()
    # Device 'mps' to use Apple Silicon GPU. Remove to use CPU
    results = model(image, stream=True, device="mps")

    for r in results:
        boxes = np.array(r.boxes.xyxy.cpu(), dtype="int")
        classes = np.array(r.boxes.cls.cpu(), dtype="int")
        confes = np.array(r.boxes.conf.cpu())
        for cls, box, conf in zip(classes, boxes, confes):
            cls_name = class_names[cls]
            if conf < min_confidence:
                continue
            conf_val = math.ceil((conf * 100)) / 100

            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            current_array = np.array([x1, y1, x2, y2, conf])
            cvzone.cornerRect(
                image,
                (x1, y1, w, h),
                l=10,
                colorR=class_colors[cls],
                t=2,
                rt=5,
            )

            cvzone.putTextRect(
                image,
                f"{cls_name} {conf_val}",
                (max(0, x1), max(30, y1)),
                scale=1,
                thickness=1,
                offset=3,
            )

    cv2.imshow("Image", image)
    cv2.waitKey(1)
