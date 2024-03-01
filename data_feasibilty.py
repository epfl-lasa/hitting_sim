import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import math
import matplotlib.pyplot as plt

from get_robot_iiwa import sim_robot




################## GET THE ROBOT ######################################

robot = sim_robot(1, 1)
robot.set_to_joint_position(robot.rest_pose)

v_dir = np.array([0, 1, 0])



################## GET THE OBJECTS ######################################

###########################################################################
# iiwa joint limits

joint_grids = []

# 35 - 1.417476e6    - 84 seconds   - 90  MB
# 40 - 8.84736e5     - 51 seconds  - 30  MB
# 45 - 3.00125e5     - 16.5 seconds  - 10  MB

grid_size = 45

sampling_interval = grid_size * math.pi / 180
mesh_size = np.array((robot.q_ul-robot.q_ll)/sampling_interval, dtype=int)

print("individual mesh sizes: ", mesh_size)
print("total data number: ", "{:e}".format(np.prod(mesh_size)))

zero_vec = [0.0] * 7
############################################################################

############################################################################
path_folder = 'data/iiwa_dataset_'+str(grid_size)

if not os.path.exists(path_folder):
    recording = 1
    os.mkdir(path_folder)

    intervals = [np.linspace(robot.q_ll[i], robot.q_ul[i], mesh_size[i]) for i in range(7)]
    
    qs = np.meshgrid(*intervals)

    mesh = np.array([qq.flatten() for qq in qs]).T
    # print(mesh.shape)
    
    N = mesh.shape[0]

    print(N)
    mesh = mesh[:N]
    q_list = []
    # EEF position
    ee_list = []
    # Mass matrices
    M_list = []
    # Inertia matrices
    Lambda_list = []
    # Inverse inertia matrices
    Lambda_inv_list = []
    # Jacobians
    J_1_list = []
    J_2_list = []
    J_3_list = []
    J_4_list = []
    J_5_list = []
    J_6_list = []
    # Position of the robot
    X_1_list = []
    X_2_list = []
    X_3_list = []
    X_4_list = []
    X_5_list = []
    X_6_list = []
else:
    recording = 0

    # load dataset directly
    mesh = np.load(path_folder+'/qs.npy')
    ts = np.load(path_folder+'/ts.npy')
    Ms = np.load(path_folder+'/Ms.npy')
    J_ts = np.load(path_folder+'/J_ts.npy')
    J_rs = np.load(path_folder+'/J_rs.npy')
    N = mesh.shape[0]


#############################################################################


st = time.time()
for joint_pos in mesh:
    if recording == 1:
        # joint_pos = robot.rest_pose
        robot.set_to_joint_position(joint_pos)
        robot.step()
        if(robot.get_self_collision_points().size == 0 and robot.get_plane_collision_points().size == 0):
            
            # Get the data
            q = robot.get_joint_position()
            Lambda = robot.get_inertia_matrix()
            Lambda_inv = robot.get_inv_inertia_matrix()
            M = robot.get_mass_matrix()

            J_1 = robot.get_trans_jacobian_point(1)
            J_2 = robot.get_trans_jacobian_point(2)
            J_3 = robot.get_trans_jacobian_point(3)
            J_4 = robot.get_trans_jacobian_point(4)
            J_5 = robot.get_trans_jacobian_point(5)
            J_6 = robot.get_trans_jacobian_point(6)

            X_1 = robot.get_point_position(1)
            X_2 = robot.get_point_position(2)
            X_3 = robot.get_point_position(3)
            X_4 = robot.get_point_position(4)
            X_5 = robot.get_point_position(5)
            X_6 = robot.get_point_position(6)


            # Store the data
            q_list.append(q)
            Lambda_list.append(Lambda)
            Lambda_inv_list.append(Lambda_inv)
            M_list.append(M)
            
            J_1_list.append(J_1)
            J_2_list.append(J_2)
            J_3_list.append(J_3)
            J_4_list.append(J_4)
            J_5_list.append(J_5)
            J_6_list.append(J_6)

            X_1_list.append(X_1)
            X_2_list.append(X_2)
            X_3_list.append(X_3)
            X_4_list.append(X_4)
            X_5_list.append(X_5)
            X_6_list.append(X_6)

        else:
            continue


np.save(path_folder + '/qs.npy', mesh)
np.save(path_folder + '/ts.npy', ts)
np.save(path_folder + '/Ms.npy', Ms)