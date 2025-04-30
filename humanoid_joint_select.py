import time
import os
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pybullet as p
import pybullet_data

###################### FUNCTION DEFINITIONS ############################

def linear_ds(A, X, X_d):
    return A @ (X - X_d)


def hitting_link(links_to_hit, box_position, human):

    '''
    Write a function for shortest distance between a link and the box'''
    distance = []
    for i in links_to_hit:  
        distance.append(np.linalg.norm(np.array(p.getLinkState(human, i+1)[4]) - np.array(box_position)))

    min_id = np.argmin(distance)

    return links_to_hit[min_id] if distance[min_id] < 0.6 else None


########################################################################

cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  cid = p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1' +
                             ' --background_color_blue=1 --width=1000 --height=1000')
p.resetDebugVisualizerCamera(cameraDistance=2.60, cameraYaw=74, cameraPitch=-34,
                                            cameraTargetPosition=[0, 0, 0])
p.resetSimulation()
useRealTime = 1
p.setTimeStep(0.01)
p.setRealTimeSimulation(useRealTime)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)


obUids = p.loadMJCF("descriptions/robot_descriptions/objects_description/objects/humanoid_fixed_monochrome.xml")
human = obUids[0]


box_left_hand = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, 0.3, 1.0], globalScaling=1.0, useFixedBase=0)


plane = p.loadURDF("plane_transparent.urdf")
p.changeVisualShape(plane, -1, rgbaColor=[0, 0, 0, 0])
p.changeDynamics(plane, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)

abdomen1 = 2
abdomen2 = 3
abdomen3 = 5

################ LEFT HAND ####################

left_hand_start_position = [-0.1, 0.3, 1.0]
left_hand_target_position = [0.8, 0.3, 1.0]

left_shoulder1 = 30
left_shoulder2 = 31
left_elbow = 33
left_wrist = 35

###################### RIGHT HAND ######################################################

right_hand_start_position = [-0.1, -0.3, 1.0]
right_hand_target_position = [0.8, -0.3, 1.0]

right_shoulder1 = 23
right_shoulder2 = 24
right_elbow = 26
right_wrist = 28

###################### LEFT LEG ########################################################

left_leg_start_position = [-0.6, 0.15, 0.1]
left_leg_target_position = [0.8, 0.15, 0.1]

left_hip1 = 15
left_hip2 = 16
left_hip3 = 17
left_knee = 19
left_ankle = 21


###################### RIGHT LEG ########################################################

right_leg_start_position = [-0.6, -0.15, 0.1]
right_leg_target_position = [0.8, -0.15, 0.1]

right_hip1 = 7
right_hip2 = 8
right_hip3 = 9
right_knee = 11
right_ankle = 13


########################################################################################


hitting_links = [10, 12, 18, 20, 25, 27, 32, 34] ## [11, 13, 19, 21, 26, 28, 33, 35]

# joint_angles = p.calculateInverseKinematics(
#     bodyUniqueId=human,
#     endEffectorLinkIndex=35,  # Link index for left_hand
#     targetPosition=left_hand_start_position,
#     solver=p.IK_DLS  # Use Damped Least Squares solver
#   )

# p.resetJointState(human, left_shoulder1, joint_angles[14])  # left_shoulder1
# p.resetJointState(human, left_shoulder2, joint_angles[15])  # left_shoulder1
# p.resetJointState(human, left_elbow, joint_angles[16])  # left_shoulder1
p.stepSimulation()

num_joints = p.getNumJoints(human)
for i in range(num_joints):
    p.changeVisualShape(human, i, rgbaColor=[0.9, 0.6, 0.5, 1])

time.sleep(5)

# ###########################################

A = np.array([[-20, 0, 0], [0, -2, 0], [0, 0, -2]])
A_leg = np.array([[-20, 0, 0], [0, -2, 0], [0, 0, -2]])

original_color = [0.9, 0.6, 0.5, 1]

previous_hit_link_id = None
current_hit_link_id = None


while(1):

    '''
    How to hit color different links in terms of what can hit the object
    '''

    box_left_hand_position = p.getBasePositionAndOrientation(box_left_hand)[0]
    
    current_hit_link_id = hitting_link(hitting_links, box_left_hand_position, human)

    print("Hit link: ", current_hit_link_id)
    if current_hit_link_id is not None:
        p.changeVisualShape(human, current_hit_link_id, rgbaColor=[1, 0, 1, 1])
        
    if previous_hit_link_id is not None and previous_hit_link_id != current_hit_link_id:
        p.changeVisualShape(human, previous_hit_link_id, rgbaColor=original_color)
    
    previous_hit_link_id = current_hit_link_id

    p.stepSimulation()
    time.sleep(0.01)