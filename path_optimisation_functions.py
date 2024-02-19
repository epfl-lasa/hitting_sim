import numpy as np
import scipy


'''Cost function'''
def vel_cost(state):
    joint_vel = state[:7]
    slack_1 = state[7:10]
    slack_2 = state[10:]
    return joint_vel.T @ joint_vel + 10*slack_1.T @ slack_1 + slack_2.T @ slack_2


def vel_cost_weight(state, weight):
    joint_vel = state[:7]
    slack_1 = state[7:10]
    slack_2 = state[10:]
    return joint_vel.T @ weight @ joint_vel + 10*slack_1.T @ slack_1 + slack_2.T @ slack_2

def fk_ineq(state, manipulator, p):
    q = state[:7]
    manipulator.set_to_joint_position(q.tolist())
    return np.linalg.norm(np.array(manipulator.get_ee_position()) - p)

def vel_ineq(state, jacobian, dx):
    joint_vel = state[:7]
    slack_1 = state[7:10]
    slack_2 = state[10:]
    # jacobian = manipulator.get_trans_jacobian_specific(joint_pos.tolist())
    
    return jacobian @ joint_vel - dx - slack_1

'''Generic Cost function'''
def vel_cost_generic(state):
    n = len(state) - 4
    joint_vel = state[:n]
    slack_1 = state[n:n+3]
    slack_2 = state[n+3:]
    return joint_vel.T @ joint_vel + 10*slack_1.T @ slack_1 + 0.5*slack_2.T @ slack_2


def vel_cost_weight_generic(state, weight):
    n = len(state) - 4
    joint_vel = state[:n]
    slack_1 = state[n:n+3]
    slack_2 = state[n+3:]
    return joint_vel.T @ weight @ joint_vel + 10*slack_1.T @ slack_1 + slack_2.T @ slack_2

def fk_ineq_generic(state, manipulator, p):
    q = state[:7]
    manipulator.set_to_joint_position(q.tolist())
    return np.linalg.norm(np.array(manipulator.get_ee_position()) - p)

def vel_ineq_generic(state, jacobian, dx):
    joint_vel = state[:7]
    slack_1 = state[7:10]
    slack_2 = state[10:]
    # jacobian = manipulator.get_trans_jacobian_specific(joint_pos.tolist())
    
    return jacobian @ joint_vel - dx - slack_1



'''
Have the desired flux at the end effector
'''

def flux_ineq(state, lambda_eff, jac, phi_des, mass_box):

    joint_vel = state[:7]
    slack_1 = state[7:10]
    slack_2 = state[10:]
    vel = jac @ joint_vel
    speed = np.linalg.norm(vel)
    flux = (lambda_eff/(lambda_eff + mass_box)) * speed  # should be a scalar

    return flux - phi_des - slack_2

'''
Alignment of the direction is also important here
'''

def dir_ineq(state, jacobian, direction):
    
    joint_pos = state[:7]
    joint_vel = state[7:14]
    vel = jacobian @ joint_pos 
    dot_p = np.dot(vel, direction)
   
    return np.abs(dot_p - 1)





'''
Functions with the state including the joint position
'''


'''Cost function'''
def vel_cost_full(state):
    joint_pos = state[:7]
    joint_vel = state[7:14]
    slack_1 = state[14:17]
    slack_2 = state[17:]
    return joint_vel.T @ joint_vel + slack_1.T @ slack_1 + slack_2.T @ slack_2


def vel_ineq_full(state, manipulator, dx):
    joint_pos = state[:7]
    joint_vel = state[7:14]
    slack_1 = state[14:17]
    slack_2 = state[17:]

    jacobian = manipulator.get_trans_jacobian_specific(joint_pos.tolist())
    
    return jacobian @ joint_vel - dx - slack_1

def vel_ineq_point_full(state, manipulator, dx, point_id):
    joint_pos = state[:7]
    joint_vel = state[7:14]
    slack_1 = state[14:17]
    slack_2 = state[17:]

    jacobian = manipulator.get_trans_jacobian_specific_point(joint_pos.tolist(), point_id)
    
    return jacobian @ joint_vel - dx - slack_1


'''
Have the desired flux at the end effector
'''

def flux_ineq_full(state, manipulator, phi_des, direction, mass_box):

    joint_pos = state[:7]
    joint_vel = state[7:14]
    slack_1 = state[14:17]
    slack_2 = state[17:]    
    
    jacobian = manipulator.get_trans_jacobian_specific(joint_pos.tolist())
    lambda_eff = manipulator.get_effective_inertia(direction)


    vel = jacobian @ joint_vel
    speed = np.linalg.norm(vel)
    flux = (lambda_eff/(lambda_eff + mass_box)) * speed  # should be a scalar

    return flux - phi_des - slack_2

def flux_ineq_point_full(state, manipulator, phi_des, direction, mass_box, point_id):

    joint_pos = state[:7]
    joint_vel = state[7:14]
    slack_1 = state[14:17]
    slack_2 = state[17:]    
    
    jacobian = manipulator.get_trans_jacobian_specific_point(joint_pos.tolist(), point_id)
    lambda_eff = manipulator.get_effective_inertia_point(direction, point_id)


    vel = jacobian @ joint_vel
    speed = np.linalg.norm(vel)
    flux = (lambda_eff/(lambda_eff + mass_box)) * speed  # should be a scalar

    return flux - phi_des - slack_2



'''
maximize the inertia of the robot
and a constraint to keep the robot behind the box
'''

def max_inertia(state_not_hit, state_hit, manipulator, direction, ee_id):
    state = np.concatenate((state_hit, state_not_hit))
    state_list = state.tolist()
    return manipulator.get_inverse_effective_inertia_specific_point(state_list, direction, ee_id)


def dot_product_constraint(state, a_1, a_2):
    return -1*np.dot(a_1, a_2)


def hit_constraints_function(state_not_hit, state_hit, manipulator, direction, ee_id):
    cons = []
    state = np.concatenate((state_hit, state_not_hit))
    state_list = state.tolist()
    
    # ee_id is the hitting one
    multi_link_pos = manipulator.get_multi_joint_position(range(len(state_list)))

    for i in range(1, len(state_not_hit)):
        print(i)
        vec = -np.array(multi_link_pos[ee_id]) + np.array(multi_link_pos[ee_id + i])
        cons.append({"type": "ineq", "fun": dot_product_constraint, "args": (vec, direction)})

    return cons
