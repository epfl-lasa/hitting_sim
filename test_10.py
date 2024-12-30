import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt


from get_robot_env_10 import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_point_NS_10, get_joint_velocities_qp
import functions as f
from path_optimisation_functions import flux_ineq_10, vel_ineq_10, vel_cost_weight, vel_cost_weight_generic, max_inertia, hit_constraints_function, dot_product_constraint

################## GET THE ROBOT ######################################
box = object.Box([0.4, 0.4, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass

robot = sim_robot_env(1, box, 1)
robot.set_to_joint_position(robot.rest_pose)

robot.ee_id = 9

##################### DS PROPERTIES ####################################
A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])

###################### DESIRED DIRECTIONAL PROPERTIES ##################
box_position_orientation = robot.get_box_position_orientation()
box_position_init = box_position_orientation[0]
box_orientation_init = box_position_orientation[1]
X_ref = f.des_hitting_point(box, box_position_init) # This needs to come from the box position
X_ref_grid = f.des_hitting_point_grid(box, box_position_init, 0, 5)

v_dir = np.array([0, 1, 0])
phi_des = 0.8


################### OPTIMIZATION FOR TOTAL DIRECTIONAL INERTIA ##############################
'''
joint limits of the robot are one source of constraints
No other constraints are considered
'''
start_pose = X_ref - np.array([0, 0.5, 0])
des_pose = robot.get_IK_joint_position_point(start_pose, robot.ee_id)
des_pose = np.array(des_pose)

robot.set_to_joint_position(des_pose)
print("des pose ", des_pose)


q_current = np.array(robot.get_joint_position())

state_hit = q_current[:robot.ee_id]
state_not_hit = q_current[robot.ee_id:]

print("state hit ", state_hit)
print("state not hit ", state_not_hit)

not_hit_ul = robot.q_ul[robot.ee_id :]
not_hit_ll = robot.q_ll[robot.ee_id :]

hit_constraints = hit_constraints_function(state_not_hit, state_hit, robot, v_dir, robot.ee_id)

print(hit_constraints)

hit_decision_variables_bound = scipy.optimize.Bounds(np.array([*not_hit_ll]), np.array([*not_hit_ul]))

hit_res = scipy.optimize.minimize(max_inertia, state_not_hit, args=(state_hit, robot, v_dir, robot.ee_id), method='SLSQP',
                                    constraints=hit_constraints,
                                    bounds=hit_decision_variables_bound,
                                    options={'disp': True})

hit_sol = hit_res.x
hit_sol = hit_sol.tolist()

des_pose[robot.ee_id:] = hit_sol

print("des pose ", des_pose)
robot.set_to_joint_position(des_pose)
robot.step()

while(1):
    des_pose = des_pose + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000, 0.0001])
    robot.set_to_joint_position(des_pose)
    robot.step()
    # time.sleep(1)
    # break

