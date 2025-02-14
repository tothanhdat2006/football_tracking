import os

import argparse
import cv2
import supervision as sv
from inference import get_model

from utils.load_dataset import get_roboflow_api_key
from configs.config_track import DetectionConfig

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video", type=str, default="./datasets/video1.mp4")
    return parser.parse_args()

def get_next_frame(video_path: str) -> cv2.VideoCapture:
    frame_gen = sv.get_video_frames_generator(video_path)
    return next(frame_gen)

def get_detections(model, frame: cv2.VideoCapture):
    result = model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    return detections

def detect_and_annotate(model, video_path: str) -> cv2.VideoCapture:
    # Annotator for 2 team players and referees
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#00BFFF', '#FF1493', '#FFD700']),
        text_color=sv.Color.from_hex('#000000'),
        text_position=sv.Position.BOTTOM_CENTER
    )

    # Annotator for ball
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex('#FFD700'),
        base=25,
        height=25,
        outline_thickness=1
    )

    tracker = sv.ByteTrack()
    tracker.reset()

    frame = get_next_frame(video_path)
    all_detections = get_detections(model, frame)
    ball_detections = []
    person_detections = []
    for detection in all_detections:
        if detection.class_id == DetectionConfig.BALL_ID:
            ball_detections.append(detection)
        else:
            person_detections.append(detection)
    
    # Perform NMS on players and referees detections and update ByteTrack
    person_detections = person_detections.with_nms(threshold=0.5, class_agnostic=True)
    person_detections.class_id -= 1
    person_detections = tracker.update_with_detections(detections=person_detections)
    labels = [
        f'#{tracker_id}' for tracker_id in person_detections.tracker_id
    ]

    annotated_frame = frame.copy()
    annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)
    annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=person_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=person_detections, labels=labels)

    return annotated_frame
    

if __name__ == "__main__":
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
    args = get_args()
    DetectionConfig.SOURCE_VIDEO_PATH = args.video
    print("Get model...")
    ROBOFLOW_API_KEY = get_roboflow_api_key()
    PLAYER_DETECTION_MODEL = get_model(model_id=DetectionConfig.PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    
    print("Detect...")
    annotated_frame = detect_and_annotate(PLAYER_DETECTION_MODEL, DetectionConfig.SOURCE_VIDEO_PATH)
    sv.plot_image(annotated_frame)
