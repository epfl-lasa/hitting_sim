import numpy as np
import pybullet as p
from get_robot_iiwa import sim_robot
import time


joint_pos_6 = np.array([0.035, 0.945, -0.156, -1.612, 1.667, -1.510, 0.955])
orientation_6 = np.array([0.707, 0, 0, 0.707])
orientation_7 = np.array([-0.707, 0, 0, 0.707])

joint_pos_7 = np.array([-0.48, 1.04, 0.02, -1.35, -1.92, -1.81, -0.81])

X_des = np.array([0.3, 0.1, 0.2])

robot = sim_robot(0, 1)

q = robot.get_IK_joint_position_orientation(X_des, orientation_6)

while(1):
    robot.set_to_joint_position(q)
    inertia_6 = robot.get_effective_inertia_point(np.array([0, 1, 0]), 6)
    print(inertia_6)
