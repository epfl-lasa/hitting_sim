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


X_des = np.array([0.3, 0.3, 0.3])

robot = sim_robot(0, 1)

num_joints = 7

###################################################

v_dir = np.array([0, 1, 0])

grid = 100

m_obj = np.linspace(0.1, 5, grid)