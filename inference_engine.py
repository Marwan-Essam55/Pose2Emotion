import torch
import torch.nn as nn
import cv2
import mediapipe as mp
import numpy as np
import json

class EmotionTransformer(nn.Module):
    def __init__(self, input_size=1533, num_classes=8):
        super(EmotionTransformer, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=3, batch_first=True)
        self.transformer = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.fc = nn.Linear(input_size, num_classes)
        
    def forward(self, x):
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

EMOTIONS = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]

def load_trained_model(weights_path, device):
    model = EmotionTransformer(input_size=1533, num_classes=8).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()
    return model

def process_video_for_fusion(video_path, model, device):
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(refine_face_landmarks=True)
    
    cap = cv2.VideoCapture(video_path)
    sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        
        pose = np.array([[res.x, res.y, res.z] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*3)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(478*3)
        
        sequence.append(np.concatenate([pose, face]))
        if len(sequence) == 40: break
    
    cap.release()
    
    if len(sequence) < 40:
        sequence.extend([np.zeros(1533)] * (40 - len(sequence)))
        
    input_tensor = torch.tensor([sequence], dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
    return {
        "label": EMOTIONS[np.argmax(probs)],
        "confidence": float(np.max(probs)),
        "all_probabilities": {EMOTIONS[i]: float(probs[i]) for i in range(len(EMOTIONS))},
        "feature_vector": input_tensor.cpu().numpy()[0][-1].tolist()
    }

# Example Usage:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# my_model = load_trained_model("emotion_transformer_1533.pth", device)
# result = process_video_for_fusion("video_sample.mp4", my_model, device)
# print(json.dumps(result, indent=4))