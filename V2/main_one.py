import json
import os
import tkinter as tk
from tkinter import filedialog
import numpy as np

from pose_extractor import extract_pose_sequence
from pose_normalizer import normalize_pose
from temporal_features import *
from behavioral_metrics import *

# Level Classification

def classify_level(value, low_th, high_th):
    if value < low_th:
        return "Low"
    elif value < high_th:
        return "Mid"
    else:
        return "High"

# Window Analysis

def analyze_windows(sequence, fps=30, window_seconds=2, overlap=0.5):
    window_size = int(window_seconds * fps)
    stride = int(window_size * (1 - overlap))

    results = []

    for start in range(0, len(sequence) - window_size, stride):
        end = start + window_size
        window_seq = sequence[start:end]

        velocity = compute_velocity(window_seq)
        acceleration = compute_acceleration(velocity)

        energy = compute_motion_energy(velocity) / len(window_seq)
        variance = compute_motion_variance(velocity)
        sudden = sudden_movement_index(acceleration)
        instability = posture_instability(window_seq)
        symmetry = symmetry_score(window_seq)

        window_result = {
            "window_time_sec": f"{round(start/fps,1)} - {round(end/fps,1)}",

            "movement_energy": {
                "value": float(energy),
                "level": classify_level(energy, 1.5, 4)
            },

            "motion_variance": {
                "value": float(variance),
                "level": classify_level(variance, 0.02, 0.07)
            },

            "sudden_movement_index": {
                "value": float(sudden),
                "level": classify_level(sudden, 0.2, 0.5)
            },

            "posture_instability": {
                "value": float(instability),
                "level": classify_level(instability, 0.2, 0.6)
            },

            "symmetry_score": {
                "value": float(symmetry),
                "level": classify_level(symmetry, 0.2, 0.5)
            }
        }

        results.append(window_result)

    return results

# File Picker

def select_video():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select a Video File",
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    return file_path

# Main

if __name__ == "__main__":

    print("\n📂 Choose a video file...")
    video_path = select_video()

    if not video_path:
        print("❌ No file selected.")
        exit()

    print("\n⏳ Extracting pose sequence...")
    sequence = extract_pose_sequence(video_path)

    if len(sequence) < 10:
        print("❌ Video too short.")
        exit()

    sequence = normalize_pose(sequence)

    print("📊 Running window-based analysis...\n")

    results = analyze_windows(sequence)

    print(json.dumps(results, indent=4))