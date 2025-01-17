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

robot.ee_id = 7

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
    robot.set_to_joint_position(des_pose)
    robot.step()
    time.sleep(1)
    break


###################### OPTIMIZATION ####################################

ul = np.concatenate((robot.q_dot_ul, 0.05*np.ones(3), 0.1*np.ones(1)))
ll = np.concatenate((robot.q_dot_ll, -0.05*np.ones(3), -0.1*np.ones(1)))

decision_variables_bound = scipy.optimize.Bounds(np.array([*ll]), np.array([*ul]))

###########################################
is_hit = False
lambda_eff = robot.get_effective_inertia_point(v_dir, robot.ee_id)

lambda_des_list = []
lambda_eff_list = []

###########################################

printed = 0

joint_vel = np.zeros(10)
slack_1 = np.zeros(3)
slack_2 = np.zeros(1)
state = np.concatenate((joint_vel, slack_1, slack_2))

q_current = np.array(robot.get_joint_position())
weight = robot.get_effective_inertia_point_influence_matrix(v_dir, robot.ee_id)

time.sleep(15)

q_dot_zeros = np.zeros(10)

while 1:
    X_qp = np.array(robot.get_point_position(robot.ee_id))
    jac = np.array(robot.get_trans_jacobian_point(robot.ee_id))

    # print("jacobian size: ", jac)
    

    '''Follow the Hitting DS and then the Linear DS'''
    if not is_hit:
        dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, v_dir, phi_des, lambda_eff, box.mass)
        hit_dir = dX / np.linalg.norm(dX)

        weight = robot.get_effective_inertia_point_influence_matrix(hit_dir, robot.ee_id)

        constraints_opt = [{"type": "eq", "fun": vel_ineq_10, "args": [jac, dX]},
                            {"type": "eq", "fun": flux_ineq_10, "args": [lambda_eff, jac, phi_des, box.mass]}]

        res = scipy.optimize.minimize(vel_cost_weight_generic, state, args=(weight), method='SLSQP',
                                    bounds=decision_variables_bound, constraints=constraints_opt,
                                    options={'disp': False})
        
        sol = res.x
        sol = sol.tolist()        
        joint_vel = sol[:10]

        joint_vel = np.array(joint_vel)
        joint_vel[robot.ee_id : ] = q_dot_zeros[robot.ee_id : ]

        joint_pos = q_current + joint_vel*0.001

        lambda_des = robot.get_effective_inertia_specific_point(joint_pos.tolist(), hit_dir, robot.ee_id)

        # lambda_des = 4.0
        lambda_des_list.append(lambda_des)
        lambda_eff_list.append(lambda_eff)
  
        q_dot = get_joint_velocities_qp_dir_inertia_specific_point_NS_10(dX, jac, robot, v_dir, 1.0, lambda_eff, lambda_des, robot.ee_id)
        q_dot[robot.ee_id : ] = q_dot_zeros[robot.ee_id : ]
        # print(q_dot)
        
        robot.move_with_joint_velocities(q_dot)
        robot.step()
    else:
        print("Hit")
        q_dot = np.zeros(10)
        robot.move_with_joint_velocities(q_dot)
        robot.step()


    # '''The different DS are controlled differently'''
    
    lambda_eff = robot.get_effective_inertia_point(hit_dir, robot.ee_id)
    q_current = np.array(robot.get_joint_position())
    state = np.concatenate((q_dot, slack_1, slack_2))


    # Need something more here later, this is contact detection and getting the contact point
    if(robot.get_collision_points().size != 0):
        is_hit = True

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
plt.title('Desired and achieved inertia (Joint 7)', fontsize=20)
plt.legend(['Desired', 'Achieved'], fontsize=16)
plt.tick_params(axis='both', labelsize=16)

# Display the plot
plt.show()