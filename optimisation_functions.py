import numpy as np
import scipy


'''Cost function'''
def vel_cost(state):
    q_dot = state[7:]
    return q_dot.T @ q_dot

def fk_ineq(state, manipulator, p):
    q = state[:7]
    manipulator.set_to_joint_position(q.tolist())
    return np.linalg.norm(np.array(manipulator.get_ee_position()) - p)

def vel_ineq(state, manipulator, v):
    q = state[:7]
    q_dot = state[7:]
    jacobian = np.array(manipulator.get_trans_jacobian_specific(q.tolist()))
    return np.square(np.linalg.norm(jacobian @ q_dot - v))

def inertia_dist(state, manipulator, l_init, direction):
    q = state[:7]
    l = manipulator.get_effective_inertia_specific(q.tolist(), direction)
    return np.abs(l - l_init)

def vel_inertia_cost(state, manipulator, l_init, direction):
    return vel_cost(state) + inertia_dist(state, manipulator, l_init, direction)

'''
Have the desired flux at the end effector
'''

def flux_ineq(state, manipulator, phi_des, direction, mass_box):

    q = state[:7]
    q_dot = state[7:]
    inv_inertia = manipulator.get_inv_inertia_matrix_specific(q.tolist())
    jacobian = np.array(manipulator.get_trans_jacobian_specific(q.tolist()))
    vel = jacobian @ q_dot
    speed = np.linalg.norm(vel)
    effective_inertia = 1/(direction.T @ inv_inertia @ direction)
    flux = (effective_inertia/(effective_inertia + mass_box)) * speed  # should be a scalar

    return np.abs(flux - phi_des)

'''
Alignment of the direction is also important here
'''

def dir_ineq(state, manipulator, direction):
    
    q = state[:7]
    q_dot = state[7:]
    jacobian = np.array(manipulator.get_trans_jacobian_specific(q.tolist()))
    vel = jacobian @ q_dot 
    cross_p = np.cross(vel, direction)
   
    return np.linalg.norm(cross_p)

