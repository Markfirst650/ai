from ultralytics import YOLO

model = YOLO("yolov8n.pt")

model.train(   
    data = "yolov8.yaml",
    workers = 0,
    epochs = 80,
    batch = 2
)
 