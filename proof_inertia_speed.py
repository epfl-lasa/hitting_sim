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
robot.ee_id = 5
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

for robot.ee_id in range(5, 4, -1):
    j = 0
    for i in np.linspace(robot.q_ll[robot.ee_id], robot.q_ul[robot.ee_id], 100):
        robot.set_to_joint_position(des_pose)
        robot.step()
        lambdas[j, robot.ee_id] = robot.get_effective_inertia_point(v_dir, robot.ee_id)
        time.sleep(0.1)
        des_pose[robot.ee_id] = i
        j += 1
# print(lambdas)

# Plot the inertia of the robot at each joint
fig, ax = plt.subplots()
for i in range(num_joints):
    ax.plot(np.linspace(robot.q_ll[i], robot.q_ul[i], 100), lambdas[:, i], label="joint " + str(i))

plt.show()