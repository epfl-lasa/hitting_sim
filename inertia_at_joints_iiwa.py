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
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
import functions as f
from path_optimisation_functions import max_inertia

################## GET THE ROBOT ######################################
box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass

robot = sim_robot_env(1, box, 1)
robot.set_to_joint_position(robot.rest_pose)

##################### DS PROPERTIES ####################################
A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])

###################### DESIRED DIRECTIONAL PROPERTIES ##################
box_position_orientation = robot.get_box_position_orientation()
box_position_init = box_position_orientation[0]
box_orientation_init = box_position_orientation[1]
X_ref = f.des_hitting_point(box, box_position_init)

X_ref[1] = -0.4
X_ref[2] = 0.5
v_dir = np.array([0, 1, 0])
phi_des = 0.7

###########################################
is_hit = False
lambda_eff = robot.get_effective_inertia(v_dir)

###########################################

robot.ee_id = 3
q_current = np.array(robot.get_joint_position())

################### OPTIMIZATION FOR TOTAL DIRECTIONAL INERTIA ##############################
'''
joint limits of the robot are one source of constraints
No other constraints are considered
'''
des_pose = robot.rest_pose

state_hit = q_current[:robot.ee_id]
state_not_hit = q_current[robot.ee_id:]

ul = robot.q_ul[robot.ee_id:]
ll = robot.q_ll[robot.ee_id:]

print(ul)
print(ll)

decision_variables_bound = scipy.optimize.Bounds(np.array([*ll]), np.array([*ul]))

res = scipy.optimize.minimize(max_inertia, state_not_hit, args=(state_hit, robot, v_dir, robot.ee_id), method='SLSQP',
                                    bounds=decision_variables_bound,
                                    options={'disp': False})

sol = res.x
sol = sol.tolist()

des_pose[robot.ee_id:] = sol

robot.step()
# robot.draw_point([robot.get_joint_cartesian_position(6)], [[0, 0, 1]], 100, 0)


while 1:
    robot.set_to_joint_position(des_pose)
    robot.step()