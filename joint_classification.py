import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt

from get_robot_iiwa import sim_robot
from iiwa_environment import object
from iiwa_environment import physics as phys

import pickle


############################### Import the gmm model ##############################
gmm_5 = pickle.load(open("gmm_models/gmm_5", "rb"))
gmm_6 = pickle.load(open("gmm_models/gmm_6", "rb"))



X_des = np.array([0.5, 0.4, 0.2])

# robot = sim_robot(0, 1)

# num_joints = 7

# ###################################################

# v_dir = np.array([0, 1, 0])

grid = 100

m_obj = np.linspace(0.1, 5, grid)
print(m_obj)