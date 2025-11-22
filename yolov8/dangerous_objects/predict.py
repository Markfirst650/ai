from ultralytics import YOLO

yolo = YOLO("best.pt",task = "detect")

result = yolo(source =r"D:\Users\Jay\Downloads\PixPin_2025-11-15_00-55-32.png",show = True ,save = True,imgsz = 1024)