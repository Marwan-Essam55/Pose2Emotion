import cv2
import numpy as np
import os
import pandas as pd
import mediapipe as mp

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5,
    refine_face_landmarks=True
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "Video_Speech_Actor_01", "Actor_01")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "Extracted_Features")

if not os.path.exists(OUTPUT_FOLDER): 
    os.makedirs(OUTPUT_FOLDER)

EMOTIONS_MAP = {
    "01": "Neutral", "02": "Calm", "03": "Happy", "04": "Sad",
    "05": "Angry", "06": "Fearful", "07": "Disgust", "08": "Surprised"
}

video_files = [f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(".mp4")]

print(f"Source Directory: {INPUT_FOLDER}")
print(f"Found {len(video_files)} videos. Starting feature extraction...")

data_list = []

for video_name in video_files:
    video_path = os.path.join(INPUT_FOLDER, video_name)
    
    try:
        parts = video_name.split("-")
        emotion_code = parts[2]
        label = EMOTIONS_MAP.get(emotion_code, "Unknown")
    except Exception:
        label = "Unknown"
        emotion_code = "01"

    cap = cv2.VideoCapture(video_path)
    all_frames_landmarks = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        if results.pose_landmarks:
            pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten()
        else:
            pose = np.zeros(33*3)

        if results.face_landmarks:
            face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
        else:
            face = np.zeros(468*3)
        
        combined = np.concatenate([pose, face])
        all_frames_landmarks.append(combined)
        
    cap.release()
    
    if len(all_frames_landmarks) > 0:
        npy_name = video_name.replace(".mp4", ".npy")
        npy_path = os.path.join(OUTPUT_FOLDER, npy_name)
        np.save(npy_path, np.array(all_frames_landmarks))
        
        data_list.append({
            "npy_path": npy_name, 
            "label": label, 
            "label_id": int(emotion_code) - 1
        })
        print(f"Successfully processed: {video_name} (Features: {combined.shape[0]})")

df = pd.DataFrame(data_list)
df.to_csv("training_data.csv", index=False)

print("\n" + "="*50)
print(f"Total files processed: {len(data_list)}")
print(f"Feature vector size: 1503 points per frame")
print(f"Output saved to: {OUTPUT_FOLDER} and training_data.csv")
print("="*50)