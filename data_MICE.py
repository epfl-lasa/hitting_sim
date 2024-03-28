import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import math
import matplotlib.pyplot as plt

from get_robot_iiwa import sim_robot
import pandas
import pickle

################## GET THE ROBOT ######################################

robot = sim_robot(0, 0)
robot.set_to_joint_position(robot.rest_pose)

v_dir = np.array([0, 1, 0])

################## GET THE OBJECTS ######################################

path_folder = '/home/harshit/developments/Robot_joint_data/'
PAD = "processedActualData"
PAD = os.path.join(path_folder, PAD + ".csv")

data = pandas.read_csv(PAD, header=0)
print("data read successfully.")

save_folder = 'data/iiwa_dataset_mice'

if not os.path.exists(save_folder):
    os.mkdir(save_folder)

joint_pos = data.iloc[:, 1:8].to_numpy()


q_list = []

M_list = []

Lambda_inv = {}
Lambda_inv['2'] = []
Lambda_inv['3'] = []
Lambda_inv['4'] = []
Lambda_inv['5'] = []
Lambda_inv['6'] = []

Jacobians = {}
Jacobians['2'] = []
Jacobians['3'] = []
Jacobians['4'] = []
Jacobians['5'] = []
Jacobians['6'] = []


Joint_X_pos = {}
Joint_X_pos['2'] = []
Joint_X_pos['3'] = []
Joint_X_pos['4'] = []
Joint_X_pos['5'] = []
Joint_X_pos['6'] = []

j = 0

start = time.time()
for i in joint_pos:
    j = j + 1
    print(j)
    robot.set_to_joint_position(i)   
    
    Lambda_inv_2 = robot.get_inv_inertia_matrix_point(2)
    Lambda_inv_3 = robot.get_inv_inertia_matrix_point(3)
    Lambda_inv_4 = robot.get_inv_inertia_matrix_point(4)
    Lambda_inv_5 = robot.get_inv_inertia_matrix_point(5)
    Lambda_inv_6 = robot.get_inv_inertia_matrix_point(6)

    M = robot.get_mass_matrix()
    
    J_2 = robot.get_trans_jacobian_point(2)
    J_3 = robot.get_trans_jacobian_point(3)
    J_4 = robot.get_trans_jacobian_point(4)
    J_5 = robot.get_trans_jacobian_point(5)
    J_6 = robot.get_trans_jacobian_point(6)

    
    X_2 = robot.get_point_position(2)
    X_3 = robot.get_point_position(3)
    X_4 = robot.get_point_position(4)
    X_5 = robot.get_point_position(5)
    X_6 = robot.get_point_position(6)


    # Store the data
    q_list.append(i)
    
    Lambda_inv['2'].append(Lambda_inv_2)
    Lambda_inv['3'].append(Lambda_inv_3)
    Lambda_inv['4'].append(Lambda_inv_4)
    Lambda_inv['5'].append(Lambda_inv_5)
    Lambda_inv['6'].append(Lambda_inv_6)

    M_list.append(M)
    
    
    Jacobians['2'].append(J_2)
    Jacobians['3'].append(J_3)
    Jacobians['4'].append(J_4)
    Jacobians['5'].append(J_5)
    Jacobians['6'].append(J_6)

    
    Joint_X_pos['2'].append(X_2)
    Joint_X_pos['3'].append(X_3)
    Joint_X_pos['4'].append(X_4)
    Joint_X_pos['5'].append(X_5)
    Joint_X_pos['6'].append(X_6)    


q_list = np.array(q_list)

Lambda_inv['2'] = np.array(Lambda_inv['2'])
Lambda_inv['3'] = np.array(Lambda_inv['3'])
Lambda_inv['4'] = np.array(Lambda_inv['4'])
Lambda_inv['5'] = np.array(Lambda_inv['5'])
Lambda_inv['6'] = np.array(Lambda_inv['6'])

M_list = np.array(M_list)


Jacobians['2'] = np.array(Jacobians['2'])
Jacobians['3'] = np.array(Jacobians['3'])
Jacobians['4'] = np.array(Jacobians['4'])
Jacobians['5'] = np.array(Jacobians['5'])
Jacobians['6'] = np.array(Jacobians['6'])


Joint_X_pos['2'] = np.array(Joint_X_pos['2'])
Joint_X_pos['3'] = np.array(Joint_X_pos['3'])
Joint_X_pos['4'] = np.array(Joint_X_pos['4'])
Joint_X_pos['5'] = np.array(Joint_X_pos['5'])
Joint_X_pos['6'] = np.array(Joint_X_pos['6'])


np.save(save_folder + '/q_list.npy', q_list)
np.save(save_folder + '/M_list.npy', M_list)

with open(save_folder + '/Joint_X_pos.pkl', 'wb') as f:
    pickle.dump(Joint_X_pos, f)

with open(save_folder + '/Lambda_inv.pkl', 'wb') as f:
    pickle.dump(Lambda_inv, f)

with open(save_folder + '/Jacobians.pkl', 'wb') as f:
    pickle.dump(Jacobians, f)    


end = time.time()
print("Time taken for the data collection: ", end - start)
#############################################################################
