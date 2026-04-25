from ultralytics import YOLO
import cv2

# Load model
model = YOLO("yolov8n.pt")

# Carnivore list
carnivores = ["lion", "tiger", "bear", "wolf", "leopard", "hyena"]

def detect_animals(image):
    results = model(image)
    carnivore_count = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if name in carnivores:
                color = (0, 0, 255)
                carnivore_count += 1
            else:
                color = (0, 255, 0)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return image, carnivore_count
