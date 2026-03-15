import numpy as np

def compute_velocity(sequence):
    return np.diff(sequence, axis=0)

def compute_acceleration(velocity):
    return np.diff(velocity, axis=0)

def compute_motion_energy(velocity):
    return np.sum(velocity ** 2)

def compute_motion_variance(velocity):
    return np.var(velocity)