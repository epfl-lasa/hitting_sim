import numpy as np
import os
import time
import scipy
import math
import matplotlib.pyplot as plt

import pickle


# ################## GET THE ROBOT ######################################
v_dir = np.array([0, 1, 0])

# ################## GET THE OBJECTS ######################################

path_folder = 'data/iiwa_dataset_mice'

joint_pos = np.load(path_folder+'/q_list.npy')

N = joint_pos.shape[0] # number of data points


with open(path_folder + '/Jacobians.pkl', 'rb') as f:
    Jacobians = pickle.load(f)


J2 = Jacobians['2']
J3 = Jacobians['3']
J4 = Jacobians['4']
J5 = Jacobians['5']
J6 = Jacobians['6']

q_dot_max = np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])
q_dot_min = -np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])

v_max = {}
v_max['2'] = []
v_max['3'] = []
v_max['4'] = []
v_max['5'] = []
v_max['6'] = []


print(J2.shape)

for i in range(N):
    print(i)

    v_max_2 = np.abs(J2[i, 1, :]) @ q_dot_max
    v_max_3 = np.abs(J3[i, 1, :]) @ q_dot_max
    v_max_4 = np.abs(J4[i, 1, :]) @ q_dot_max
    v_max_5 = np.abs(J5[i, 1, :]) @ q_dot_max
    v_max_6 = np.abs(J6[i, 1, :]) @ q_dot_max

    v_max['2'].append(v_max_2)
    v_max['3'].append(v_max_3)
    v_max['4'].append(v_max_4)
    v_max['5'].append(v_max_5)
    v_max['6'].append(v_max_6)

 
v_max['2'] = np.array(v_max['2'])
v_max['3'] = np.array(v_max['3'])
v_max['4'] = np.array(v_max['4'])
v_max['5'] = np.array(v_max['5'])
v_max['6'] = np.array(v_max['6'])

with open(path_folder + '/v_max.pkl', 'wb') as f:
    pickle.dump(v_max, f)


   
