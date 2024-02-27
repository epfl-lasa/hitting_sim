import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
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

sampling_interval = grid_size
mesh_size = np.array((robot.q_ul-robot.q_ul)/sampling_interval, dtype=int)

print("individual mesh sizes: ", mesh_size)
print("total data number: ", "{:e}".format(np.prod(mesh_size)))

zero_vec = [0.0] * 7
############################################################################

############################################################################
path_folder = 'data/iiwa_dataset'+str(grid_size)

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
    # EEF position
    ts = np.zeros((N, 3))
    # Mass matrices
    Ms = np.zeros((N, 7, 7))
    Lambda_pos = np.zeros((N, 3, 3))
    # Jacobians
    J_ts = np.zeros((N, 3, 7))
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
for i in range(N):
    if recording == 1:
        break

np.save(path_folder + '/qs.npy', mesh)
np.save(path_folder + '/ts.npy', ts)
np.save(path_folder + '/Ms.npy', Ms)