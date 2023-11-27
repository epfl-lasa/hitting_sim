import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy

from get_robot_env import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
import functions as f
from path_optimisation_functions import flux_ineq, vel_cost, vel_ineq

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
X_ref = f.des_hitting_point(box, box_position_init) # This needs to come from the box position
X_ref_grid = f.des_hitting_point_grid(box, box_position_init, 0, 5)

v_dir = np.array([0, 1, 0])
phi_des = 0.7

###################### OPTIMIZATION ####################################

ul = np.concatenate((robot.q_dot_ul, 0.05*np.ones(3), 0.1*np.ones(1)))
ll = np.concatenate((robot.q_dot_ll, -0.05*np.ones(3), -0.1*np.ones(1)))

decision_variables_bound = scipy.optimize.Bounds(np.array([*ll]), np.array([*ul]))

###########################################
is_hit = False
lambda_eff = robot.get_effective_inertia(v_dir)

full_inertia = robot.get_inertia_matrix()

###########################################

printed = 0

f.get_closest_joint(robot, X_ref)

robot.ee_id = 6

joint_vel = np.zeros(7)
slack_1 = np.zeros(3)
slack_2 = np.zeros(1)
state = np.concatenate((joint_vel, slack_1, slack_2))
print("state ", state.shape)

q_current = np.array(robot.get_joint_position())

while 1:
    X_qp = np.array(robot.get_ee_position())
    jac = np.array(robot.get_trans_jacobian())

    '''Follow the Hitting DS and then the Linear DS'''
    if not is_hit:

        
        dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, v_dir, phi_des, lambda_eff, box.mass)
        hit_dir = dX / np.linalg.norm(dX)


        constraints_opt = [{"type": "eq", "fun": vel_ineq, "args": [jac, dX]},
                            {"type": "eq", "fun": flux_ineq, "args": [lambda_eff, jac, phi_des, box.mass]}]

        res = scipy.optimize.minimize(vel_cost, state, method='SLSQP',
                                    bounds=decision_variables_bound, constraints=constraints_opt,
                                    options={'disp': False})
        sol = res.x
        sol = sol.tolist()        
        joint_vel = sol[:7]

        joint_vel = np.array(joint_vel)

        # q_dot = joint_vel

        joint_pos = q_current + joint_vel*0.001
        lambda_des = robot.get_effective_inertia_specific(joint_pos.tolist(), hit_dir)
        q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, robot, hit_dir, 0.15, lambda_eff, lambda_des)

        print("inertia matrix: ", robot.get_inertia_matrix())

    else:
        dX = linear_ds(A, X_qp, X_ref)
        q_dot = get_joint_velocities_qp(dX, jac, robot)


    # '''The different DS are controlled differently'''
    robot.move_with_joint_velocities(q_dot)

    # Need something more here later, this is contact detection and getting the contact point
    if(robot.get_collision_points().size != 0):
        is_hit = True
        robot.get_collision_position()
        hit_point = robot.get_collision_position()
        hit_velocity = robot.get_ee_velocity_current()
        hit_inertia = robot.get_effective_inertia(hit_dir)
        hit_joint_pos = robot.get_joint_position()
    
    if printed == 0 and is_hit:
        print("hit point ", hit_point)
        print("hit velocity ", hit_velocity)
        print("hit inertia ", hit_inertia)
        print("hit joint pos ", hit_joint_pos)
        printed = 1


    robot.step()

    lambda_eff = robot.get_effective_inertia(hit_dir)
    q_current = np.array(robot.get_joint_position())

    
    state = np.concatenate((q_dot, slack_1, slack_2))

