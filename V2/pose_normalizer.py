import numpy as np

def normalize_pose(sequence):
    normalized = []

    for frame in sequence:
        left_hip = frame[23]
        right_hip = frame[24]
        center = (left_hip + right_hip) / 2

        frame_centered = frame - center

        left_shoulder = frame[11]
        right_shoulder = frame[12]
        shoulder_dist = np.linalg.norm(left_shoulder - right_shoulder)

        if shoulder_dist > 0:
            frame_scaled = frame_centered / shoulder_dist
        else:
            frame_scaled = frame_centered

        normalized.append(frame_scaled)

    return np.array(normalized)