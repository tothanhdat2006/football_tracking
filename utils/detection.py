from typing import List
import numpy as np

from tqdm import tqdm
import cv2
import supervision as sv

from configs.config import DetectionConfig 

def get_next_frame(video_path: str) -> cv2.VideoCapture:
    frame_gen = sv.get_video_frames_generator(video_path)
    return next(frame_gen)

def get_detections(model, frame: cv2.VideoCapture) -> sv.Detections:
    result = model.infer(frame, confidence=0.3)[0]
    detections = sv.Detections.from_inference(result)
    return detections

def get_keypoints(model, frame: cv2.VideoCapture) -> sv.KeyPoints:
    result = model.infer(frame, confidence=0.3)[0]
    keypoints = sv.KeyPoints.from_inference(result)
    return keypoints

def get_crops(model, source_video: str) -> List[np.ndarray]:
    frame_generator = sv.get_video_frames_generator(source_path=source_video, stride=DetectionConfig.STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc="Collecting crops"):
        detections = get_detections(model, frame)
        detections = detections.with_nms(threshold=0.5, class_agnostic=True)
        detections = detections[detections.class_id == DetectionConfig.PLAYER_ID]
        player_crops = [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]
        crops += player_crops

    return crops