import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import math
import matplotlib.pyplot as plt

from get_robot_iiwa import sim_robot
import pickle

################## GET THE ROBOT ######################################

robot = sim_robot(0, 0)
robot.set_to_joint_position(robot.rest_pose)

v_dir = np.array([0, 1, 0])

################## GET THE OBJECTS ######################################

###########################################################################
# iiwa joint limits

joint_grids = []

# 35 - 1.417476e6    - 84 seconds   - 90  MB
# 40 - 8.84736e5     - 51 seconds  - 30  MB
# 45 - 3.00125e5     - 16.5 seconds  - 10  MB

grid_size = 15

grid_size_array = np.array([30, 30, 30, 45, 45, 45, 90])

sampling_interval = grid_size_array * math.pi / 180
mesh_size = np.array((robot.q_ul-robot.q_ll)/sampling_interval, dtype=int)

print("individual mesh sizes: ", mesh_size)
print("total data number: ", "{:e}".format(np.prod(mesh_size)))

zero_vec = [0.0] * 7
############################################################################

############################################################################
path_folder = 'data/iiwa_dataset_plane_adapt_'+str(grid_size)

if not os.path.exists(path_folder):
    recording = 1
    os.mkdir(path_folder)

    intervals = [np.linspace(robot.q_ll[i], robot.q_ul[i], mesh_size[i]) for i in range(7)]
    
    qs = np.meshgrid(*intervals)

    mesh = np.array([qq.flatten() for qq in qs]).T
    # print(mesh.shape)
    
    N = mesh.shape[0]

    print("size = ", N)
    mesh = mesh[:N]
    q_list = []
    
    # Mass matrices
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

    q_list_collision = []

    M_list_collision = []

    Lambda_inv_collision = {}
    Lambda_inv_collision['2'] = []
    Lambda_inv_collision['3'] = []
    Lambda_inv_collision['4'] = []
    Lambda_inv_collision['5'] = []
    Lambda_inv_collision['6'] = []

    Jacobians_collision = {}
    Jacobians_collision['2'] = []
    Jacobians_collision['3'] = []
    Jacobians_collision['4'] = []
    Jacobians_collision['5'] = []
    Jacobians_collision['6'] = []


    Joint_X_pos_collision = {}
    Joint_X_pos_collision['2'] = []
    Joint_X_pos_collision['3'] = []
    Joint_X_pos_collision['4'] = []
    Joint_X_pos_collision['5'] = []
    Joint_X_pos_collision['6'] = []
    
    
else:
    recording = 0

    # load dataset directly
    mesh = np.load(path_folder+'/qs.npy')
    ts = np.load(path_folder+'/ts.npy')
    Ms = np.load(path_folder+'/Ms.npy')
    J_ts = np.load(path_folder+'/J_ts.npy')
    J_rs = np.load(path_folder+'/J_rs.npy')
    N = mesh.shape[0]


# #############################################################################

i = 0
j = 0

st = time.time()
for joint_pos in mesh:
    if recording == 1:
        robot.set_to_joint_position(joint_pos)
        # robot.step()
        # if(robot.get_self_collision_points().size == 0 and robot.get_plane_collision_points().size == 0):
        if(robot.get_plane_collision_points().size == 0):
            i = i + 1
            print("all good", i)      
            # Get the data
            q = robot.get_joint_position()
            
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
            q_list.append(q)
            
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

            # time.sleep(0.1)        
            
        else:
            j = j + 1

            print("Collision", j)
            continue

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


np.save(path_folder + '/q_list.npy', q_list)
np.save(path_folder + '/M_list.npy', M_list)

with open(path_folder + '/Joint_X_pos.pkl', 'wb') as f:
    pickle.dump(Joint_X_pos, f)

with open(path_folder + '/Lambda_inv.pkl', 'wb') as f:
    pickle.dump(Lambda_inv, f)

with open(path_folder + '/Jacobians.pkl', 'wb') as f:
    pickle.dump(Jacobians, f)

