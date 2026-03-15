import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_full.task"

def extract_pose_sequence(video_path):
    BaseOptions = python.BaseOptions
    PoseLandmarker = vision.PoseLandmarker
    PoseLandmarkerOptions = vision.PoseLandmarkerOptions
    VisionRunningMode = vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO
    )

    sequence = []

    with PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                keypoints = np.array(
                    [[lm.x, lm.y, lm.z] for lm in result.pose_landmarks[0]]
                )
            else:
                keypoints = np.zeros((33, 3))

            sequence.append(keypoints)
            frame_idx += 1

        cap.release()

    return np.array(sequence)