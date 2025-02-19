import os
import torch

import matplotlib.pyplot as plt
import argparse
import cv2
import supervision as sv
from inference import get_model

from configs.config import DetectionConfig, PitchConfig
from utils.load_dataset import get_roboflow_api_key
from utils.detection import *
from utils.draw import *
from utils.view import ViewTransformer
from utils.team import TeamClassifier, resolve_goalkeepers_team_id

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", help="Path to video", type=str, default="./datasets/video1.mp4")
    return parser.parse_args()

def detect_and_annotate(PLAYER_DETECTION_MODEL, frame: cv2.VideoCapture, team_classifier: TeamClassifier, tracker: sv.ByteTrack) -> cv2.VideoCapture | List[sv.Detections]:
    '''
    Detect and annotate players, goalkeepers and referees on the frame

    Args:
        PLAYER_DETECTION_MODEL: Model for detecting players, goalkeepers and referees
        frame (cv2.VideoCapture): Frame to detect and annotate
        team_classifier (TeamClassifier): Classifier for classifying team of players
        tracker (sv.ByteTrack): Tracker for tracking players, goalkeepers and referees
    
    Returns:
        cv2.VideoCapture: Annotated frame
        List[sv.Detections]: and detections of players, goalkeepers and referees
    '''
    # Annotator for players
    ellipse_annotator = sv.EllipseAnnotator(
        color=sv.ColorPalette.from_hex(['#20b0bc', '#f04889', '#FFD700']),
        thickness=2
    )

    label_annotator = sv.LabelAnnotator(
        color=sv.ColorPalette.from_hex(['#20b0bc', '#f04889', '#FFD700']),
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

    # Get detections for each objects
    all_detections = get_detections(PLAYER_DETECTION_MODEL, frame)

    ball_detections = all_detections[all_detections.class_id == DetectionConfig.BALL_ID]
    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

    person_detections = all_detections[all_detections.class_id != DetectionConfig.BALL_ID]
    person_detections = person_detections.with_nms(threshold=0.5, class_agnostic=True)
    person_detections = tracker.update_with_detections(detections=person_detections)

    goalkeepers_detections = person_detections[person_detections.class_id == DetectionConfig.GOALKEEPER_ID]
    players_detections = person_detections[person_detections.class_id == DetectionConfig.PLAYER_ID]
    referees_detections = person_detections[person_detections.class_id == DetectionConfig.REFEREE_ID]
    
    # Classifiy team of players
    players_crops = [sv.crop_image(frame, xyxy) for xyxy in players_detections.xyxy]
    players_detections.class_id = team_classifier.predict(players_crops)

    goalkeepers_detections.class_id = resolve_goalkeepers_team_id(players_detections, goalkeepers_detections)

    referees_detections.class_id -= 1

    # Merge goalkeepers, players and referees detections
    person_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections, referees_detections
    ])
    person_detections.class_id = person_detections.class_id.astype(int)
    
    labels = [
        f'#{tracker_id}' for tracker_id in person_detections.tracker_id
    ]

    annotated_frame = frame.copy()
    annotated_frame = triangle_annotator.annotate(scene=annotated_frame, detections=ball_detections)
    annotated_frame = ellipse_annotator.annotate(scene=annotated_frame, detections=person_detections)
    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=person_detections, labels=labels)

    return annotated_frame, [ball_detections, players_detections, goalkeepers_detections, referees_detections]
    
def draw_pitch_map(FIELD_DETECTION_MODEL, frame: cv2.VideoCapture, detections: List[sv.Detections]) -> np.ndarray:
    '''
    Draw map of players on pitch

    Args:
        FIELD_DETECTION_MODEL: Model for detecting pitch
        frame (cv2.VideoCapture): Frame to detect and annotate
        detections (List[sv.Detections]): Detections of players, goalkeepers and referees

    Returns:
        np.ndarray: 2D representation of map of players on pitch
    '''
    ball_detections, players_detections, goalkeepers_detections, referees_detections = detections
    # Create map of players by transforming points from frame to pitch
    team_detections = sv.Detections.merge([
        players_detections, goalkeepers_detections
    ])

    keypoints = get_keypoints(FIELD_DETECTION_MODEL, frame)

    CONFIG = PitchConfig()
    filter = keypoints.confidence[0] > 0.5
    # Get points in transformation matrix
    frame_reference_points = keypoints.xy[0][filter]
    pitch_reference_points = np.array(CONFIG.vertices)[filter]

    # Create transformation matrix
    transformer = ViewTransformer(
        source=frame_reference_points,
        target=pitch_reference_points
    )

    frame_ball_xy = ball_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_ball_xy = transformer.transform_points(points=frame_ball_xy)
    
    frame_team_xy = team_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_team_xy = transformer.transform_points(points=frame_team_xy)
    
    frame_referees_xy = referees_detections.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    pitch_referees_xy = transformer.transform_points(points=frame_referees_xy)
    
    # Annotate players on pitch
    pitch_view = draw_pitch(CONFIG)
    ## Draw ball
    pitch_view = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_ball_xy,
        face_color=sv.Color.WHITE,
        edge_color=sv.Color.BLACK,
        radius=10,
        pitch=pitch_view
    )
    ## Draw team 1
    pitch_view = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_team_xy[team_detections.class_id == 0],
        face_color=sv.Color.from_hex('00BFFF'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=pitch_view
    )
    ## Draw team 2
    pitch_view = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_team_xy[team_detections.class_id == 1],
        face_color=sv.Color.from_hex('FF1493'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=pitch_view
    )
    ## Draw referee
    pitch_view = draw_points_on_pitch(
        config=CONFIG,
        xy=pitch_referees_xy,
        face_color=sv.Color.from_hex('FFD700'),
        edge_color=sv.Color.BLACK,
        radius=16,
        pitch=pitch_view
    )

    return pitch_view


if __name__ == "__main__":
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "[CUDAExecutionProvider]"
    args = get_args()
    DetectionConfig.SOURCE_VIDEO_PATH = args.video

    ROBOFLOW_API_KEY = get_roboflow_api_key()
    PLAYER_DETECTION_MODEL = get_model(model_id=DetectionConfig.PLAYER_DETECTION_MODEL_ID, api_key=ROBOFLOW_API_KEY)
    FIELD_DETECTION_MODEL = get_model(model_id=DetectionConfig.PLAYER_TRACKING_MODEL_ID, api_key=ROBOFLOW_API_KEY)

    crops = get_crops(PLAYER_DETECTION_MODEL, DetectionConfig.SOURCE_VIDEO_PATH)
    team_classifier = TeamClassifier(device="cuda")
    team_classifier.fit(crops)
        
    # frame = get_next_frame(DetectionConfig.SOURCE_VIDEO_PATH)
    cap = cv2.VideoCapture(DetectionConfig.SOURCE_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    annotated_writer = cv2.VideoWriter(
        './outputs/annotated_output.mp4',
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height)
    )
    pitch_writer = None

    frame_generator = sv.get_video_frames_generator(DetectionConfig.SOURCE_VIDEO_PATH)
    tracker = sv.ByteTrack()
    tracker.reset()
    for frame in tqdm(frame_generator, desc="Processing frames"):
        annotated_frame, detections = detect_and_annotate(PLAYER_DETECTION_MODEL, frame, team_classifier, tracker)
        pitch_view = draw_pitch_map(FIELD_DETECTION_MODEL, frame, detections)

        annotated_writer.write(annotated_frame)

        if pitch_writer is None:
            pitch_writer = cv2.VideoWriter(
                './outputs/pitch_output.mp4',
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                pitch_view.shape[:2][::-1]
            )

        pitch_writer.write(pitch_view)

    annotated_writer.release()
    if pitch_writer is not None:
        pitch_writer.release()
    # print("Detecting and drawing pitch...")
    # cv2.imwrite('./outputs/output_frame.png', annotated_frame)
    # cv2.imwrite('./outputs/output_map.png', pitch_view)
