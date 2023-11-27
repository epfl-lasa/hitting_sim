import numpy as np
import os
import time
import pybullet as p
import pybullet_data
from qpsolvers import solve_ls
import scipy
from get_robot import sim_robot


'''Cost function'''
def vel_cost(state):
    q_dot = state[7:]
    return q_dot.T @ q_dot

''''''

# State is joint position and joint velocities

def fk_ineq(state, manipulator, p):
    q = state[:7]
    
    manipulator.set_to_joint_position(q.tolist())
    
    return np.linalg.norm(np.array(manipulator.get_ee_position()) - p)

def flux_ineq(state, manipulator, phi_des, direction, mass_box):

    q = state[:7]
    q_dot = state[7:]

    manipulator.set_to_joint_position(q.tolist())

    inv_inertia = manipulator.get_inv_inertia_matrix_specific(q.tolist())
    jacobian = np.array(manipulator.get_trans_jacobian_specific(q.tolist()))
    vel = jacobian @ q_dot
    speed = np.linalg.norm(vel)

    effective_inertia = 1/(direction.T @ inv_inertia @ direction)
    flux = (effective_inertia/(effective_inertia + mass_box)) * vel  # should be a scalar

    return np.abs(flux - phi_des)

def dir_ineq(state, manipulator, direction):
    
    q = state[:7]
    q_dot = state[7:]
    manipulator.set_to_joint_position(q.tolist())
    jacobian = np.array(manipulator.get_trans_jacobian_specific(q.tolist()))
    vel = jacobian @ q_dot 
    cross_p = np.cross(vel, direction)
   
    return np.linalg.norm(cross_p)

# Constraint of forward kinematics

# Change this where you would like the box to be at
t_des = np.array([0.58, 0, 0.3])

################## GET THE ROBOT ######################################
robot = sim_robot(1)
robot.set_to_joint_position(robot.rest_pose)

###################### DESIRED DIRECTIONAL PROPERTIES ##################

v_dir = np.array([0, 1, 0])

# joint position limit

ul = np.concatenate((robot.q_ul, robot.q_dot_ul))
ll = np.concatenate((robot.q_ll, robot.q_dot_ll))

decision_variables_bound = scipy.optimize.Bounds(np.array([*ll]), np.array([*ul]))

# prepare initial guess
state_0 = np.concatenate((robot.rest_pose, np.zeros(7)))
print ("state_0", state_0)


constraints_opt = [{"type": "eq", "fun": fk_ineq, "args": [robot, t_des]},
                   {"type": "eq", "fun": flux_ineq, "args": [robot, 0.5, v_dir, 0.5]},
                   {"type": "eq", "fun": dir_ineq, "args": [robot, v_dir]}]

res = scipy.optimize.minimize(vel_cost, state_0, method='SLSQP',
                              bounds=decision_variables_bound, constraints=constraints_opt,
                              options={'disp': True})
print("res fun", res)


sol = res.x
sol = sol.tolist()

print("sol", sol)
printed = 0
while 1:
    robot.set_to_joint_position(sol)

    if printed == 0:
        print("ee pos: ", robot.get_ee_position())
        printed = 1
    p.stepSimulation()
