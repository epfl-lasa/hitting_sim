import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt

from get_robot_env import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
import functions as f
from path_optimisation_functions import flux_ineq, vel_cost, vel_ineq, vel_cost_weight

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

robot.ee_id = 6

q_current = np.array(robot.get_joint_position())

'''
I wanna see which joints can reach the desired point
through IK
'''

q_des = robot.get_IK_joint_position_point(X_ref, robot.ee_id)

# q_des = robot.rest_pose
# q_des = robot.get_IK_joint_position(X_ref)
robot.set_to_joint_position(q_des)
robot.step()
print(robot.get_ee_position())

robot.draw_point([X_ref.tolist()], [[1, 0, 0]], 50, 0)
while 1:
    # robot.set_to_joint_position(q_des)
    
    # q_des[5] = q_des[5] + 0.1
    # robot.step()
    robot.set_to_joint_position(q_des)
    
    # print(robot.get_effective_inertia_point(v_dir, robot.ee_id))
    robot.step()
    # print(robot.get_effective_inertia_point_influence_matrix(v_dir, robot.ee_id))
    # time.sleep(1)
    # i+=1
    # dq[robot.ee_id + 1] = 0.05







# while 1:
#     robot.set_to_joint_position(robot.rest_pose)
#     print(robot.get_effective_inertia(v_dir))

#     robot.step()
