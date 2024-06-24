from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.export(format="engine")