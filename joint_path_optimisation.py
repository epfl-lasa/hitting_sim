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

lambda_des_list = []
lambda_eff_list = []

###########################################

printed = 0

f.get_closest_joint(robot, X_ref)

robot.ee_id = 6

joint_vel = np.zeros(7)
slack_1 = np.zeros(3)
slack_2 = np.zeros(1)
state = np.concatenate((joint_vel, slack_1, slack_2))

q_current = np.array(robot.get_joint_position())
weight = robot.get_effective_inertia_influence_matrix(v_dir)

while 1:
    X_qp = np.array(robot.get_ee_position())
    jac = np.array(robot.get_trans_jacobian())
    

    '''Follow the Hitting DS and then the Linear DS'''
    if not is_hit:

        
        dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, v_dir, phi_des, lambda_eff, box.mass)
        hit_dir = dX / np.linalg.norm(dX)


        constraints_opt = [{"type": "eq", "fun": vel_ineq, "args": [jac, dX]},
                            {"type": "eq", "fun": flux_ineq, "args": [lambda_eff, jac, phi_des, box.mass]}]

        res = scipy.optimize.minimize(vel_cost_weight, state, args=(weight), method='SLSQP',
                                    bounds=decision_variables_bound, constraints=constraints_opt,
                                    options={'disp': False})
        
        # res = scipy.optimize.minimize(vel_cost, state, method='SLSQP',
        #                             bounds=decision_variables_bound, constraints=constraints_opt,
        #                             options={'disp': False})
        
        sol = res.x
        sol = sol.tolist()        
        joint_vel = sol[:7]

        joint_vel = np.array(joint_vel)

        # q_dot = joint_vel

        joint_pos = q_current + joint_vel*0.001
        lambda_des = robot.get_effective_inertia_specific(joint_pos.tolist(), hit_dir)
        # lambda_des = 5
        lambda_des_list.append(lambda_des)
        lambda_eff_list.append(lambda_eff)
        q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, robot, hit_dir, 0.15, lambda_eff, lambda_des)

        # print(robot.get_effective_inertia(hit_dir), 1/robot.get_inverse_effective_inertia(hit_dir))
        # print(robot.get_inverse_effective_inertia_gradient(hit_dir), robot.get_effective_inertia_gradient(hit_dir))
        # print("matrix: ", robot.get_effective_inertia_influence_matrix(hit_dir))
        print("lambda eff ", lambda_eff, "lambda des ", lambda_des, "flux ", lambda_eff/(lambda_eff + box.mass)*np.linalg.norm(jac @ q_dot))

    else:
        dX = linear_ds(A, X_qp, X_ref)
        q_dot = get_joint_velocities_qp(dX, jac, robot)

    # '''The different DS are controlled differently'''
    robot.move_with_joint_velocities(q_dot)
    weight = robot.get_effective_inertia_influence_matrix(hit_dir)

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


    '''
    To plot the desired and achieved inertia, 
    uncomment the lines ahead (you will lose control of the robot)
    '''
    if is_hit and robot.get_box_speed() < 0.001:
        break

lambda_eff_list = np.array(lambda_eff_list)
lambda_des_list = np.array(lambda_des_list)

plt.plot(lambda_des_list, color='blue', marker='o', linestyle='dashed',
     linewidth=2, markersize=5)
plt.plot(lambda_eff_list, color='orange', marker='*', linestyle='dashed',
     linewidth=2, markersize=5)

# Add labels and a title
plt.xlabel('Time', fontsize=16)
plt.ylabel('Inertia', fontsize=16)
plt.title('Desired and achieved inertia', fontsize=20)
plt.legend(['Desired', 'Achieved'], fontsize=16)
plt.tick_params(axis='both', labelsize=16)

# Display the plot
plt.show()