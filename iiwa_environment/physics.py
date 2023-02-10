import numpy as np

def get_initial_speed(d, mu):
    return np.sqrt(2 * mu * 9.81 * d)