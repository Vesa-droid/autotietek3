from ultralytics import YOLO

model = YOLO("yolo-Weights/yolov8n.pt")

model.train(data="C:/Python projekti/dataset/data.yaml", epochs=20, augment=True)