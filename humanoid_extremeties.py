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

def linear_hitting_ds_pre_impact(A1, X, X_obj, v_hit, p_des, lambda_current, m_obj):
    obj_virtual = X_obj + np.dot((X - X_obj), v_hit) * v_hit / np.square(np.linalg.norm(v_hit))
    sigma = 0.1
    alpha = np.exp(-np.linalg.norm(X - obj_virtual)/np.square(sigma))
    dX = alpha * v_hit + (1 - alpha) * A1 @ (X - obj_virtual)
    dX = (p_des/lambda_current)*(lambda_current+m_obj) * dX / np.linalg.norm(dX)
    return dX

########################################################################


cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  cid = p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1' +
                             ' --background_color_blue=1 --width=1000 --height=1000')
p.resetDebugVisualizerCamera(cameraDistance=2.60, cameraYaw=20.8, cameraPitch=-33,
                                            cameraTargetPosition=[0, 0, 0])
p.resetSimulation()
useRealTime = 1
p.setTimeStep(0.01)
p.setRealTimeSimulation(useRealTime)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)


obUids = p.loadMJCF("mjcf/humanoid_fixed.xml")
human = obUids[0]


box_right_foot = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.0, -0.15, 0.1], globalScaling=1.0, useFixedBase=0)
box_left_foot = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.0, 0.15, 0.1], globalScaling=1.0, useFixedBase=0)
box_right_hand = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, -0.3, 1.0], globalScaling=1.0, useFixedBase=0)
box_left_hand = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, 0.3, 1.0], globalScaling=1.0, useFixedBase=0)
p.changeDynamics(box_right_foot, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_left_foot, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_right_hand, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_left_hand, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)                 



table_1 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.2, -0.2, -0.3], globalScaling=1.0, useFixedBase=1)
table_2 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.2, 0.2, -0.3], globalScaling=1.0, useFixedBase=1)
table_3 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.7, -0.4, 0.6], globalScaling=1.0, useFixedBase=1)
table_4 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.7, 0.4, 0.6], globalScaling=1.0, useFixedBase=1)
p.changeDynamics(table_1, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(table_2, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(table_3, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(table_4, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)


plane = p.loadURDF("plane_transparent.urdf")
p.changeVisualShape(plane, -1, rgbaColor=[0, 0, 0, 0])
p.changeDynamics(plane, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)

# Print joint and link information

# print("Joint and Link Mapping:")
# for i in range(p.getNumJoints(human)):
#     joint_info = p.getJointInfo(human, i)
#     joint_index = joint_info[0]
#     joint_name = joint_info[1].decode('utf-8')  # Joint name
#     link_name = joint_info[12].decode('utf-8')  # Child link name
#     print(f"Joint Index: {joint_index}, Joint Name: {joint_name}, Link Name: {link_name}")

# # Print base/root link information
# print("\nBase Link:")


'''
for j in range(p.getNumJoints(human)):
  # print("Joint: ", j)
  p.changeDynamics(human, j, linearDamping=0, angularDamping=0)
  info = p.getJointInfo(human, j)
  # print(info)

  jointName = info[1]
  jointType = info[2]
  print(jointName, "   ", jointType)
  # if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
  #   jointIds.append(j)
  #   paramIds.append(p.addUserDebugParameter(jointName.decode("utf-8"), -4, 4, 0))

  link_info = p.getLinkState(human, j)
  # print(len(link_info))
'''

abdomen1 = 2
abdomen2 = 3
abdomen3 = 5

################ LEFT HAND ####################

left_hand_start_position = [-0.1, 0.3, 1.0]
left_hand_target_position = [0.8, 0.3, 1.0]

left_shoulder1 = 30
left_shoulder2 = 31
left_elbow = 33

###################### RIGHT HAND ######################################################

right_hand_start_position = [-0.1, -0.3, 1.0]
right_hand_target_position = [0.8, -0.3, 1.0]

right_shoulder1 = 23
right_shoulder2 = 24
right_elbow = 26

###################### LEFT LEG ########################################################

left_leg_start_position = [-0.6, 0.15, 0.1]
left_leg_target_position = [0.8, 0.15, 0.1]

left_hip1 = 15
left_hip2 = 16
left_hip3 = 17
left_knee = 19


###################### RIGHT LEG ########################################################

right_leg_start_position = [-0.6, -0.15, 0.1]
right_leg_target_position = [0.8, -0.15, 0.1]

right_hip1 = 7
right_hip2 = 8
right_hip3 = 9
right_knee = 11


########################################################################################
# num_joints = p.getNumJoints(human)
# for i in range(num_joints):
#     p.changeVisualShape(human, i, rgbaColor=[0.9, 0.6, 0.5, 1])



joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=35,  # Link index for left_hand
    targetPosition=left_hand_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

p.resetJointState(human, left_shoulder1, joint_angles[14])  # left_shoulder1
p.resetJointState(human, left_shoulder2, joint_angles[15])  # left_shoulder1
p.resetJointState(human, left_elbow, joint_angles[16])  # left_shoulder1
p.stepSimulation()

#########################################################################################

joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=28,  # Link index for left_hand
    targetPosition=right_hand_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )


p.resetJointState(human, right_shoulder1, joint_angles[11])  # left_shoulder1
p.resetJointState(human, right_shoulder2, joint_angles[12])  # left_shoulder1
p.resetJointState(human, right_elbow, joint_angles[13])  # left_shoulder1
p.stepSimulation()

#########################################################################################
joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=21,  # Link index for left_hand
    targetPosition=left_leg_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

# p.resetJointState(human, abdomen1, joint_angles[0])  # left_shoulder1
p.resetJointState(human, abdomen2, joint_angles[1])  # left_shoulder1
# p.resetJointState(human, abdomen3, joint_angles[2])  # left_shoulder1
p.resetJointState(human, left_hip1, joint_angles[7])  # left_shoulder?1
p.resetJointState(human, left_hip2, joint_angles[8])  # left_shoulder1
p.resetJointState(human, left_hip3, joint_angles[9])  # left_shoulder1
p.resetJointState(human, left_knee, joint_angles[10])  # left_shoulder1
p.stepSimulation()

# #############################################################################################

joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=13,  # Link index for left_hand
    targetPosition=right_leg_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

# p.resetJointState(human, abdomen1, joint_angles[0])  # left_shoulder1
# p.resetJointState(human, abdomen2, joint_angles[1])  # left_shoulder1
# p.resetJointState(human, abdomen3, joint_angles[2])  # left_shoulder1
p.resetJointState(human, right_hip1, joint_angles[3])  # left_shoulder1
p.resetJointState(human, right_hip2, joint_angles[4])  # left_shoulder1
p.resetJointState(human, right_hip3, joint_angles[5])  # left_shoulder1
p.resetJointState(human, right_knee, joint_angles[6])  # left_shoulder1
p.stepSimulation()


time.sleep(5)


# ###########################################

A = np.array([[-20, 0, 0], [0, -2, 0], [0, 0, -2]])
A_leg = np.array([[-17, 0, 0], [0, -2, 0], [0, 0, -2]])

is_hit_left_hand = False
is_hit_right_hand = False
is_hit_left_foot = False




while (not is_hit_left_hand):

  left_hand_position = p.getLinkState(human, 35)[4]
  left_hand_des_vel = linear_ds(A, np.array(left_hand_position), np.array(left_hand_target_position))
  left_hand_next_position = np.array(left_hand_position) + left_hand_des_vel*0.01

  left_hand_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=35,  # Link index for left_hand
    targetPosition=left_hand_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

  p.setJointMotorControl2(human, left_shoulder1, p.POSITION_CONTROL, targetPosition=left_hand_joint_angles[14])  # left_shoulder1 
  p.setJointMotorControl2(human, left_shoulder2, p.POSITION_CONTROL, targetPosition=left_hand_joint_angles[15])  # left_shoulder1
  p.setJointMotorControl2(human, left_elbow, p.POSITION_CONTROL, targetPosition=left_hand_joint_angles[16])  # left_shoulder1

  p.performCollisionDetection()
  points_collision = np.array(p.getContactPoints(human, box_left_hand), dtype=object)


  if(points_collision.size != 0):
    is_hit_left_hand = True
    print("HIT")

  p.stepSimulation()
  time.sleep(0.01)

while (not is_hit_right_hand):

  p.setJointMotorControl2(human, left_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1 
  p.setJointMotorControl2(human, left_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1

  right_hand_position = p.getLinkState(human, 28)[4]
  right_hand_des_vel = linear_ds(A, np.array(right_hand_position), np.array(right_hand_target_position))
  right_hand_next_position = np.array(right_hand_position) + right_hand_des_vel*0.01
  right_hand_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=28,  # Link index for left_hand
    targetPosition=right_hand_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

  p.setJointMotorControl2(human, right_shoulder1, p.POSITION_CONTROL, targetPosition=right_hand_joint_angles[11])  # left_shoulder1
  p.setJointMotorControl2(human, right_shoulder2, p.POSITION_CONTROL, targetPosition=right_hand_joint_angles[12])  # left_shoulder1
  p.setJointMotorControl2(human, right_elbow, p.POSITION_CONTROL, targetPosition=right_hand_joint_angles[13])  # left
 
  p.performCollisionDetection()
  points_collision = np.array(p.getContactPoints(human, box_right_hand), dtype=object)


  if(points_collision.size != 0):
    is_hit_right_hand = True
    print("HIT")

  p.stepSimulation()
  time.sleep(0.01)

joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=21,  # Link index for left_hand
    targetPosition=left_leg_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

# # p.resetJointState(human, abdomen1, joint_angles[0])  # left_shoulder1
# p.resetJointState(human, abdomen2, joint_angles[1])  # left_shoulder1
# # p.resetJointState(human, abdomen3, joint_angles[2])  # left_shoulder1
# p.resetJointState(human, left_hip1, joint_angles[7])  # left_shoulder?1
# p.resetJointState(human, left_hip2, joint_angles[8])  # left_shoulder1
# p.resetJointState(human, left_hip3, joint_angles[9])  # left_shoulder1
# p.resetJointState(human, left_knee, joint_angles[10])  # left_shoulder1
# p.stepSimulation()

# # #############################################################################################

# joint_angles = p.calculateInverseKinematics(
#     bodyUniqueId=human,
#     endEffectorLinkIndex=13,  # Link index for left_hand
#     targetPosition=right_leg_start_position,
#     solver=p.IK_DLS  # Use Damped Least Squares solver
#   )

# # p.resetJointState(human, abdomen1, joint_angles[0])  # left_shoulder1
# p.resetJointState(human, abdomen2, joint_angles[1])  # left_shoulder1
# # p.resetJointState(human, abdomen3, joint_angles[2])  # left_shoulder1
# p.resetJointState(human, right_hip1, joint_angles[3])  # left_shoulder1
# p.resetJointState(human, right_hip2, joint_angles[4])  # left_shoulder1
# p.resetJointState(human, right_hip3, joint_angles[5])  # left_shoulder1
# p.resetJointState(human, right_knee, joint_angles[6])  # left_shoulder1
# p.stepSimulation()


# time.sleep(2)

# while(1):
#   p.stepSimulation()
#   time.sleep(0.01)   


while (not is_hit_left_foot):

  p.setJointMotorControl2(human, left_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1 
  p.setJointMotorControl2(human, left_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1

  p.setJointMotorControl2(human, right_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1




  left_foot_position = p.getLinkState(human, 21)[4]
  left_foot_des_vel = linear_ds(A_leg, np.array(left_foot_position), np.array(left_leg_target_position))
  left_foot_next_position = np.array(left_foot_position) + left_foot_des_vel*0.01


  right_foot_position = p.getLinkState(human, 13)[4]
  right_foot_des_vel = linear_ds(A_leg, np.array(right_foot_position), np.array(right_leg_target_position))
  right_foot_next_position = np.array(right_foot_position) + right_foot_des_vel*0.01

  left_foot_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=21,  # Link index for left_hand
    targetPosition=left_foot_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

  right_foot_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=13,  # Link index for left_hand
    targetPosition=right_foot_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  ) 

  p.setJointMotorControl2(human, abdomen2, p.POSITION_CONTROL, targetPosition=left_foot_joint_angles[1])  # left_shoulder1
  p.setJointMotorControl2(human, left_hip1, p.POSITION_CONTROL, targetPosition=left_foot_joint_angles[7])  # left_shoulder1
  p.setJointMotorControl2(human, left_hip2, p.POSITION_CONTROL, targetPosition=left_foot_joint_angles[8])  # left_shoulder1
  p.setJointMotorControl2(human, left_hip3, p.POSITION_CONTROL, targetPosition=left_foot_joint_angles[9])  # left_shoulder1
  p.setJointMotorControl2(human, left_knee, p.POSITION_CONTROL, targetPosition=left_foot_joint_angles[10])  # left_shoulder1


  p.setJointMotorControl2(human, right_hip1, p.POSITION_CONTROL, targetPosition=right_foot_joint_angles[3])  # left_shoulder1
  p.setJointMotorControl2(human, right_hip2, p.POSITION_CONTROL, targetPosition=right_foot_joint_angles[4])  # left_shoulder1
  p.setJointMotorControl2(human, right_hip3, p.POSITION_CONTROL, targetPosition=right_foot_joint_angles[5])  # left_shoulder1
  p.setJointMotorControl2(human, right_knee, p.POSITION_CONTROL, targetPosition=right_foot_joint_angles[6])  # left_shoulder1



  p.performCollisionDetection()
  points_collision = np.array(p.getContactPoints(human, box_left_foot), dtype=object)

  if(points_collision.size != 0):
    is_hit_left_foot = True
    print("HIT")



  p.stepSimulation()
  time.sleep(0.01)

while (1):

  p.setJointMotorControl2(human, left_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1 
  p.setJointMotorControl2(human, left_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1

  p.setJointMotorControl2(human, right_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1

  p.setJointMotorControl2(human, abdomen2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_hip1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_hip2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_hip3, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_knee, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1

  p.setJointMotorControl2(human, right_hip1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_hip2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_hip3, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_knee, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1



  p.stepSimulation()
  time.sleep(0.01)


