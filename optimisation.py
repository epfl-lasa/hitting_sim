import numpy as np
import os
import time
import pybullet as p
import pybullet_data
from qpsolvers import solve_ls
import scipy
from get_robot_env import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
import functions as f


from optimisation_functions import fk_ineq, flux_ineq, dir_ineq, vel_cost, total_cost


################## GET THE ROBOT ######################################
box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass
test_robot = sim_robot_env(1, box, 0)
test_robot.set_to_joint_position(test_robot.rest_pose)

robot = sim_robot_env(1, box, 1)
robot.set_to_joint_position(robot.rest_pose)
# robot.step()

##################### DS PROPERTIES ####################################
A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])

###################### DESIRED DIRECTIONAL PROPERTIES ##################
box_position_orientation = robot.get_box_position_orientation()
box_position_init = box_position_orientation[0]
box_orientation_init = box_position_orientation[1]
X_ref = f.des_hitting_point(box, box_position_init) # This needs to come from the box position
X_ref_grid = f.des_hitting_point_grid(box, box_position_init, 0, 5)

# X_ref = np.array([0.5, 0.3, 0.5])
v_dir = np.array([0, 1, 0])
phi_des = 0.5

###################### OPTIMIZATION ####################################
# joint position limit

ul = np.concatenate((test_robot.q_ul, test_robot.q_dot_ul))
ll = np.concatenate((test_robot.q_ll, test_robot.q_dot_ll))

decision_variables_bound = scipy.optimize.Bounds(np.array([*ll]), np.array([*ul]))

# prepare initial guess
state_0 = np.concatenate((test_robot.rest_pose, np.zeros(7,)))

constraints_opt = [{"type": "eq", "fun": fk_ineq, "args": [test_robot, X_ref]},
                   {"type": "eq", "fun": flux_ineq, "args": [test_robot, phi_des, v_dir, 0.5]},
                   {"type": "eq", "fun": dir_ineq, "args": [test_robot, v_dir]}]

# res = scipy.optimize.minimize(total_cost, state_0, args=(robot.rest_pose), method='SLSQP',
#                               bounds=decision_variables_bound, constraints=constraints_opt,
#                               options={'disp': True})
res = scipy.optimize.minimize(vel_cost, state_0, method='SLSQP',
                              bounds=decision_variables_bound, constraints=constraints_opt,
                              options={'disp': True})

print("res fun", res)

sol = res.x
sol = sol.tolist()

joint_q = sol[:7]
joint_q_dot = sol[7:]
print("sol", sol)
printed = 0


############################################################

lambda_ref = robot.get_effective_inertia_specific(joint_q, v_dir)
speed_ref = np.array(robot.get_trans_jacobian_specific(joint_q)) @ np.array(joint_q_dot)

print("lambda ref ", lambda_ref)
print("speed ref ", speed_ref)

###########################################
is_hit = False
lambda_eff = robot.get_effective_inertia(v_dir)

###########################################

printed = 0

f.get_closest_joint(robot, X_ref)

while 1:
    # robot.set_to_joint_position(joint_q)

    X_qp = np.array(robot.get_ee_position())

    # '''Follow the Hitting DS and then the Linear DS'''
    if not is_hit:
        dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, v_dir, phi_des, lambda_eff, box.mass)
    else:
        dX = linear_ds(A, X_qp, X_ref)

    hit_dir = dX / np.linalg.norm(dX)

    lambda_eff = robot.get_effective_inertia(hit_dir)
    jac = np.array(robot.get_trans_jacobian())

    # '''The different DS are controlled differently'''
    if not is_hit:
        q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, robot, hit_dir, 0.15, lambda_eff, lambda_ref)
    else:
        q_dot = get_joint_velocities_qp(dX, jac, robot)
    
    
    robot.move_with_joint_velocities(q_dot)

    # ## Need something more here later, this is contact detection and getting the contact point
    if(robot.get_collision_points().size != 0):
        is_hit = True
        robot.get_collision_position()
        hit_point = robot.get_collision_position()
        hit_velocity = robot.get_ee_velocity_current()
        hit_inertia = robot.get_effective_inertia(hit_dir)
    
    if printed == 0 and is_hit:
        print("hit point ", hit_point)
        print("hit velocity ", hit_velocity)
        print("hit inertia ", hit_inertia)
        printed = 1


    robot.step()
    # time_now = time.time()


    # printed = 1
    # p.stepSimulation()