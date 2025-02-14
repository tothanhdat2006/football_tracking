from ultralytics import YOLO
from utils.load_dataset import load_dataset

if __name__ == "__name__":
    model = YOLO("yolov8x-pose.yaml")
    dataset, project = load_dataset()
    results = model.train(task='detect', model='yolo8x-pose.pt', data=f'{dataset.location}/data.yaml', epochs=50, imgsz=1280, batch=8)
    project.version(dataset.version).deploy(model_type='yolov8-pose', model_path='./runs/pose/train/')
    