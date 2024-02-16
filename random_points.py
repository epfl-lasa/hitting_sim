import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt

from get_robot_env_iiwa import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_point_NS, get_joint_velocities_qp
import functions as f
from path_optimisation_functions import flux_ineq, vel_ineq, vel_cost_weight, vel_cost_weight_generic, max_inertia

################## GET THE ROBOT ######################################
box = object.Box([0.3, 0.3, 0.3], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass

robot = sim_robot_env(1, box, 1)
robot.set_to_joint_position(robot.rest_pose)

#Robot ee id can be changed here
robot.ee_id = 1


################### OPTIMIZATION FOR TOTAL DIRECTIONAL INERTIA ##############################
'''
joint limits of the robot are one source of constraints
No other constraints are considered
'''

des_pose = np.zeros(7)

robot.set_to_joint_position(des_pose)

start = time.time()

# while time.time() - start < 0.1:
for i in range(7):
    robot.step()

    print(robot.get_joint_cartesian_position(robot.ee_id))

