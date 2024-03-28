import numpy as np
import os
import time
import pybullet as p
import pybullet_data
import scipy
import matplotlib.pyplot as plt

from get_robot_iiwa import sim_robot
from iiwa_environment import object
from iiwa_environment import physics as phys

'''
The role of this code:
We look at how the robot's inertia changes at a joint when the not hitting joint moves
'''

################## GET THE ROBOT ######################################

X_des = np.array([0.3, 0.3, 0.3])

robot = sim_robot(0, 1)

num_joints = 7

robot.draw_point([X_des], [[1, 0, 0]], 30, 0)

###################### DESIRED DIRECTIONAL PROPERTIES ##################
v_dir = np.array([0, 1, 0])

grid = 100

lambdas = np.zeros((grid, num_joints))
speeds = np.zeros((grid, num_joints))
fluxes = np.zeros((grid, num_joints))

range_ee = range(6, 3, -1)

decision_bounds = np.stack((robot.q_dot_ll, robot.q_dot_ul))
decision_bounds = np.transpose(decision_bounds)


for robot.ee_id in range_ee:
    des_pose = np.array(robot.get_IK_joint_position(X_des))

    j = 0
    for i in np.linspace(robot.q_ll[robot.ee_id], robot.q_ul[robot.ee_id], 100):
        des_pose[robot.ee_id] = i
        robot.set_to_joint_position(des_pose)
        robot.step()
        lambdas[j, robot.ee_id] = robot.get_effective_inertia_point(v_dir, robot.ee_id)
        J = np.array(robot.get_trans_jacobian_point(robot.ee_id))
        q_dot_speed = scipy.optimize.linprog(-1*np.absolute(J[1,:]), bounds= decision_bounds)
        
        speed = np.array(robot.get_trans_jacobian_point(robot.ee_id)) @ q_dot_speed.x
        speeds[j, robot.ee_id] = speed[1] 
        
        fluxes[j, robot.ee_id] = (lambdas[j, robot.ee_id] / (lambdas[j, robot.ee_id] + 2)) * speeds[j, robot.ee_id]
        
        time.sleep(0.05)
        j += 1

# Plot the inertia of the robot at each joint
fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
for i in range_ee:
    ax0.plot(np.linspace(robot.q_ll[i], robot.q_ul[i], 100), lambdas[:, i], label="joint " + str(i), linewidth=5)
    ax1.plot(np.linspace(robot.q_ll[i], robot.q_ul[i], 100), speeds[:, i], label="joint " + str(i), linewidth=5)
    ax2.plot(np.linspace(robot.q_ll[i], robot.q_ul[i], 100), fluxes[:, i], label="joint " + str(i), linewidth=5)


# ax0.set_title("Inertia", fontsize=16)
ax0.set_xlabel("Joint motion range (rad)", fontsize=10)
ax0.set_ylabel("Directional Inertia", fontsize=10)
ax0.legend()

# ax1.set_title("Speed", fontsize=16)
ax1.set_xlabel("Joint motion range (rad)", fontsize=10)
ax1.set_ylabel("Speed", fontsize=10)
ax1.legend()

# ax2.set_title("Hitting Flux", fontsize=16)
ax2.set_xlabel("Joint motion range (rad)", fontsize=10)
ax2.set_ylabel("Hitting Flux", fontsize=10)
ax2.legend()

plt.show()