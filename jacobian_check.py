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

# Robot ee id can be changed here

robot.ee_id = 6 

################### OPTIMIZATION FOR TOTAL DIRECTIONAL INERTIA ##############################
'''
joint limits of the robot are one source of constraints
No other constraints are considered
'''
# des_pose = robot.rest_pose
des_pose = np.zeros(7)

q_current = np.array(robot.get_joint_position())

robot.set_to_joint_position(des_pose)
robot.step()

q_current = np.array(robot.get_joint_position())

X0 = np.array(robot.get_point_position(0))
X1 = np.array(robot.get_point_position(1))
X2 = np.array(robot.get_point_position(2))
X3 = np.array(robot.get_point_position(3))
X4 = np.array(robot.get_point_position(4))
X5 = np.array(robot.get_point_position(5))
X6 = np.array(robot.get_point_position(6))

X_0 = np.array(robot.get_joint_cartesian_position(0))
X_1 = np.array(robot.get_joint_cartesian_position(1))
X_2 = np.array(robot.get_joint_cartesian_position(2))
X_3 = np.array(robot.get_joint_cartesian_position(3))
X_4 = np.array(robot.get_joint_cartesian_position(4))
X_5 = np.array(robot.get_joint_cartesian_position(5))
X_6 = np.array(robot.get_joint_cartesian_position(6))



print(X0)
print(X1)
print(X2)
print(X3)
print(X4)
print(X5)
print(X6)


while (1):
    X0 = np.array(robot.get_point_position(0))
    X1 = np.array(robot.get_point_position(1))
    X2 = np.array(robot.get_point_position(2))
    X3 = np.array(robot.get_point_position(3))
    X4 = np.array(robot.get_point_position(4))
    X5 = np.array(robot.get_point_position(5))
    X6 = np.array(robot.get_point_position(6))

    X_0 = np.array(robot.get_joint_cartesian_position(0))
    X_1 = np.array(robot.get_joint_cartesian_position(1))
    X_2 = np.array(robot.get_joint_cartesian_position(2))
    X_3 = np.array(robot.get_joint_cartesian_position(3))
    X_4 = np.array(robot.get_joint_cartesian_position(4))
    X_5 = np.array(robot.get_joint_cartesian_position(5))
    X_6 = np.array(robot.get_joint_cartesian_position(6))
    
    jac = np.array(robot.get_trans_jacobian_point(robot.ee_id))

    robot.draw_point([X0.tolist()], [[1, 0, 0]], 20, 0)
    robot.draw_point([X1.tolist()], [[1, 0, 0]], 20, 0)
    robot.draw_point([X2.tolist()], [[1, 0, 0]], 20, 0)
    robot.draw_point([X3.tolist()], [[1, 0, 0]], 20, 0)
    robot.draw_point([X4.tolist()], [[1, 0, 0]], 20, 0)
    robot.draw_point([X5.tolist()], [[1, 0, 0]], 20, 0)
    robot.draw_point([X6.tolist()], [[1, 0, 0]], 20, 0)

    robot.draw_point([X_0.tolist()], [[0, 1, 0]], 20, 0)
    robot.draw_point([X_1.tolist()], [[0, 1, 0]], 20, 0)
    robot.draw_point([X_2.tolist()], [[0, 1, 0]], 20, 0)
    robot.draw_point([X_3.tolist()], [[0, 1, 0]], 20, 0)
    robot.draw_point([X_4.tolist()], [[0, 1, 0]], 20, 0)
    robot.draw_point([X_5.tolist()], [[0, 1, 0]], 20, 0)
    robot.draw_point([X_6.tolist()], [[0, 1, 0]], 20, 0)

