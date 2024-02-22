import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt

from get_robot_env_iiwa import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys

'''
The role of this code:
We look at how the robot's inertia changes at a joint when the not hitting joint moves
'''

################## GET THE ROBOT ######################################
box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass

robot = sim_robot_env(1, box, 1)
robot.set_to_joint_position(robot.rest_pose)

# Robot ee id can be changed here
# robot.ee_id = 5
num_joints = 7

###################### DESIRED DIRECTIONAL PROPERTIES ##################
v_dir = np.array([0, 1, 0])

des_pose = robot.rest_pose

q_current = np.array(robot.get_joint_position())

state_hit = q_current[:robot.ee_id]
state_not_hit = q_current[robot.ee_id:]

print("des pose ", des_pose)
robot.set_to_joint_position(des_pose)
robot.step()

grid = 100

lambdas = np.zeros((grid, num_joints))
speeds = np.zeros((grid, num_joints))
fluxes = np.zeros((grid, num_joints))

range_ee = range(6, 3 -1)

for robot.ee_id in range_ee:
    j = 0
    for i in np.linspace(robot.q_ll[robot.ee_id], robot.q_ul[robot.ee_id], 100):
        robot.set_to_joint_position(des_pose)
        robot.step()
        lambdas[j, robot.ee_id] = robot.get_effective_inertia_point(v_dir, robot.ee_id)
        speeds[j, robot.ee_id] = np.linalg.norm(np.array(robot.get_trans_jacobian_point(robot.ee_id)) @ robot.q_dot_ul)
        fluxes[j, robot.ee_id] = (lambdas[j, robot.ee_id] / (lambdas[j, robot.ee_id] + 2)) * speeds[j, robot.ee_id]
        time.sleep(0.1)
        des_pose[robot.ee_id] = i
        j += 1

# Plot the inertia of the robot at each joint
# fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
# for i in range_ee:
#     ax0.plot(np.linspace(robot.q_ll[i], robot.q_ul[i], 100), lambdas[:, i], label="joint " + str(i))
#     ax1.plot(np.linspace(robot.q_ll[i], robot.q_ul[i], 100), speeds[:, i], label="joint " + str(i))
#     ax2.plot(np.linspace(robot.q_ll[i], robot.q_ul[i], 100), fluxes[:, i], label="joint " + str(i))


# ax0.set_title("Inertia")
# ax0.set_xlabel("Joint position")
# ax0.set_ylabel("Inertia")
# # ax0.legend()

# ax1.set_title("Speed")
# ax1.set_xlabel("Joint position")
# ax1.set_ylabel("Speed")
# # ax1.legend()

# ax2.set_title("Flux")
# ax2.set_xlabel("Joint position")
# ax2.set_ylabel("Flux")
# # ax2.legend()

# plt.show()