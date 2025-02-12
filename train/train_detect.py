from ultralytics import YOLO
from utils.load_dataset import load_dataset

model = YOLO("yolo11x.yaml")
dataset, project = load_dataset()
results = model.train(task='detect', model='yolo11x.pt', data=f'{dataset.location}/data.yaml', epochs=50, imgsz=1280, batch=8)
project.version(dataset.version).deploy(model_type='yolo11', model_path='./runs/detect/train/')