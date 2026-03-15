import numpy as np

def sudden_movement_index(acceleration, threshold=0.02):
    sudden = np.abs(acceleration) > threshold
    return np.sum(sudden) / acceleration.size

def posture_instability(sequence):
    spine = sequence[:, 0:5, :]  # upper body approx
    return np.var(spine)

def symmetry_score(sequence):
    left = sequence[:, [11,13,15,23,25,27], :]
    right = sequence[:, [12,14,16,24,26,28], :]
    return np.mean(np.abs(left - right))