import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt

from get_robot_iiwa import sim_robot


################## GET THE ROBOT ######################################

robot = sim_robot(0, 1)
robot.set_to_joint_position(robot.rest_pose)

# Robot ee id can be changed here

robot.ee_id = 6 

################### OPTIMIZATION FOR TOTAL DIRECTIONAL INERTIA ##############################
'''
joint limits of the robot are one source of constraints
No other constraints are considered
'''
des_pose = np.array([-2.960000018166925, -1.0050702259266853e-11, -1.48, -1.5045993919859056e-08, -2.959999922790245, 2.2587981844424473e-07, 3.049999948449989])
# des_pose = np.zeros(7)

robot.set_to_joint_position(des_pose)
# robot.step()

# print(robot.get_inertia_matrix())
# robot.set_to_joint_position(des_pose)
# robot.step()
print(robot.get_inertia_matrix_point(5))


