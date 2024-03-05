import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt

from get_robot_iiwa import sim_robot


################## GET THE ROBOT ######################################

robot = sim_robot(1, 1)
robot.set_to_joint_position(robot.rest_pose)

# Robot ee id can be changed here

robot.ee_id = 6 

################### OPTIMIZATION FOR TOTAL DIRECTIONAL INERTIA ##############################
'''
joint limits of the robot are one source of constraints
No other constraints are considered
'''
des_pose = robot.rest_pose
# des_pose = np.zeros(7)

q_current = np.array(robot.get_joint_position())

robot.set_to_joint_position(des_pose)
robot.step()

q_current = np.array(robot.get_joint_position())
