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
p.resetDebugVisualizerCamera(cameraDistance=2.60, cameraYaw=12.8, cameraPitch=-43,
                                            cameraTargetPosition=[0, 0, 0])
p.resetSimulation()
useRealTime = 1
p.setTimeStep(0.01)
p.setRealTimeSimulation(useRealTime)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)


obUids = p.loadMJCF("mjcf/humanoid_fixed.xml")
human = obUids[0]


box_right_knee = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.1, -0.15, 0.55], globalScaling=1.0, useFixedBase=0)
box_left_knee = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.1, 0.15, 0.55], globalScaling=1.0, useFixedBase=0)
box_right_elbow = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [-0.3, -0.3, 1.25], globalScaling=1.0, useFixedBase=0)
box_left_elbow = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [-0.3, 0.3, 1.25], globalScaling=1.0, useFixedBase=0)
p.changeDynamics(box_right_knee, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_left_knee, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_right_elbow, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_left_elbow, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)                 



table_1 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.4, -0.2, 0.15], globalScaling=1.0, useFixedBase=1)
table_2 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.4, 0.2, 0.15], globalScaling=1.0, useFixedBase=1)
table_3 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [-0.6, -0.4, 0.85], globalScaling=1.0, useFixedBase=1)
table_4 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [-0.6, 0.4, 0.85], globalScaling=1.0, useFixedBase=1)
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


# logid = p.startStateLogging(p.STATE_LOGGING_GENERIC_ROBOT, "huamanoid_middities_hit.txt", [human])
# logid = p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "huamanoid_middities_hit.mp4", [human])

abdomen1 = 2
abdomen2 = 3
abdomen3 = 5

################ LEFT elbow ####################

left_elbow_start_position = [0.3, 0.3, 1.25]
left_elbow_target_position = [-0.9, 0.3, 1.25]

left_shoulder1 = 30
left_shoulder2 = 31
left_elbow = 33

###################### RIGHT elbow ######################################################

right_elbow_start_position = [0.3, -0.3, 1.25]
right_elbow_target_position = [-0.9, -0.3, 1.25]

right_shoulder1 = 23
right_shoulder2 = 24
right_elbow = 26

###################### LEFT LEG ########################################################

left_knee_start_position = [-0.45, 0.15, 0.55]
left_knee_target_position = [0.9, 0.15, 0.55]

left_hip1 = 15
left_hip2 = 16
left_hip3 = 17
left_knee = 19


###################### RIGHT LEG ########################################################

right_knee_start_position = [-0.45, -0.15, 0.55]
right_knee_target_position = [0.9, -0.15, 0.55]

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
    endEffectorLinkIndex=33,  # Link index for left_elbow
    targetPosition=left_elbow_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

p.resetJointState(human, left_shoulder1, joint_angles[14])  # left_shoulder1
p.resetJointState(human, left_shoulder2, joint_angles[15])  # left_shoulder1
p.resetJointState(human, left_elbow, joint_angles[16])  # left_shoulder1
p.stepSimulation()

#########################################################################################

joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=26,  # Link index for left_elbow
    targetPosition=right_elbow_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )


p.resetJointState(human, right_shoulder1, joint_angles[11])  # left_shoulder1
p.resetJointState(human, right_shoulder2, joint_angles[12])  # left_shoulder1
p.resetJointState(human, right_elbow, joint_angles[13])  # left_shoulder1
p.stepSimulation()

#########################################################################################
joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=19,  # Link index for left_elbow
    targetPosition=left_knee_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

print(joint_angles)

# right_knee_angle = joint_angles[10] + 0.5

# p.resetJointState(human, abdomen1, joint_angles[0])  # left_shoulder1
p.resetJointState(human, abdomen2, joint_angles[1])  # left_shoulder1
# p.resetJointState(human, abdomen3, joint_angles[2])  # left_shoulder1
p.resetJointState(human, left_hip1, joint_angles[7])  # left_shoulder?1
p.resetJointState(human, left_hip2, joint_angles[8])  # left_shoulder1
p.resetJointState(human, left_hip3, joint_angles[9])  # left_shoulder1
p.resetJointState(human, left_knee, joint_angles[10] - 1)  # left_shoulder1
p.stepSimulation()

# #############################################################################################

joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=11,  # Link index for left_elbow
    targetPosition=right_knee_start_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

# p.resetJointState(human, abdomen1, joint_angles[0])  # left_shoulder1
# p.resetJointState(human, abdomen2, joint_angles[1])  # left_shoulder1
# p.resetJointState(human, abdomen3, joint_angles[2])  # left_shoulder1
p.resetJointState(human, right_hip1, joint_angles[3])  # left_shoulder1
p.resetJointState(human, right_hip2, joint_angles[4])  # left_shoulder1
p.resetJointState(human, right_hip3, joint_angles[5])  # left_shoulder1
p.resetJointState(human, right_knee, joint_angles[6] - 1)  # left_shoulder1
p.stepSimulation()


time.sleep(10)


# ###########################################

A = np.array([[-20, 0, 0], [0, -2, 0], [0, 0, -2]])
A_leg = np.array([[-15, 0, 0], [0, -2, 0], [0, 0, -2]])

is_hit_left_elbow = False
is_hit_right_elbow = False   
is_hit_left_knee = False

# while(1):
#   p.stepSimulation()
#   time.sleep(0.01)   


while (not is_hit_left_elbow):

  left_elbow_position = p.getLinkState(human, 33)[4]
  left_elbow_des_vel = linear_ds(A, np.array(left_elbow_position), np.array(left_elbow_target_position))
  left_elbow_next_position = np.array(left_elbow_position) + left_elbow_des_vel*0.01

  left_elbow_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=33,  # Link index for left_elbow
    targetPosition=left_elbow_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

  p.setJointMotorControl2(human, left_shoulder1, p.POSITION_CONTROL, targetPosition=left_elbow_joint_angles[14])  # left_shoulder1 
  p.setJointMotorControl2(human, left_shoulder2, p.POSITION_CONTROL, targetPosition=left_elbow_joint_angles[15])  # left_shoulder1
  p.setJointMotorControl2(human, left_elbow, p.POSITION_CONTROL, targetPosition=left_elbow_joint_angles[16])  # left_shoulder1

  p.performCollisionDetection()
  points_collision = np.array(p.getContactPoints(human, box_left_elbow), dtype=object)


  if(points_collision.size != 0):
    is_hit_left_elbow = True
    print("HIT")

  p.stepSimulation()
  time.sleep(0.01)

while (not is_hit_right_elbow):

  p.setJointMotorControl2(human, left_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1 
  p.setJointMotorControl2(human, left_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1

  right_elbow_position = p.getLinkState(human, 26)[4]
  right_elbow_des_vel = linear_ds(A, np.array(right_elbow_position), np.array(right_elbow_target_position))
  right_elbow_next_position = np.array(right_elbow_position) + right_elbow_des_vel*0.01
  right_elbow_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=26,  # Link index for left_elbow
    targetPosition=right_elbow_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

  p.setJointMotorControl2(human, right_shoulder1, p.POSITION_CONTROL, targetPosition=right_elbow_joint_angles[11])  # left_shoulder1
  p.setJointMotorControl2(human, right_shoulder2, p.POSITION_CONTROL, targetPosition=right_elbow_joint_angles[12])  # left_shoulder1
  p.setJointMotorControl2(human, right_elbow, p.POSITION_CONTROL, targetPosition=right_elbow_joint_angles[13])  # left
 
  p.performCollisionDetection()
  points_collision = np.array(p.getContactPoints(human, box_right_elbow), dtype=object)


  if(points_collision.size != 0):
    is_hit_right_elbow = True
    print("HIT")

  p.stepSimulation()
  time.sleep(0.01)



while (not is_hit_left_knee):

  p.setJointMotorControl2(human, left_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1 
  p.setJointMotorControl2(human, left_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, left_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1

  p.setJointMotorControl2(human, right_shoulder1, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_shoulder2, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1
  p.setJointMotorControl2(human, right_elbow, p.VELOCITY_CONTROL, targetVelocity=0)  # left_shoulder1




  left_knee_position = p.getLinkState(human, 19)[4]
  left_knee_des_vel = linear_ds(A_leg, np.array(left_knee_position), np.array(left_knee_target_position))
  left_knee_next_position = np.array(left_knee_position) + left_knee_des_vel*0.01


  right_knee_position = p.getLinkState(human, 11)[4]
  right_knee_des_vel = linear_ds(A_leg, np.array(right_knee_position), np.array(right_knee_target_position))
  right_knee_next_position = np.array(right_knee_position) + right_knee_des_vel*0.01

  left_knee_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=19,  # Link index for left_elbow
    targetPosition=left_knee_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

  right_knee_joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=11,  # Link index for left_elbow
    targetPosition=right_knee_next_position,
    solver=p.IK_DLS  # Use Damped Least Squares solver
  ) 

  p.setJointMotorControl2(human, abdomen2, p.POSITION_CONTROL, targetPosition=left_knee_joint_angles[1])  # left_shoulder1
  p.setJointMotorControl2(human, left_hip1, p.POSITION_CONTROL, targetPosition=left_knee_joint_angles[7])  # left_shoulder1
  p.setJointMotorControl2(human, left_hip2, p.POSITION_CONTROL, targetPosition=left_knee_joint_angles[8])  # left_shoulder1
  p.setJointMotorControl2(human, left_hip3, p.POSITION_CONTROL, targetPosition=left_knee_joint_angles[9])  # left_shoulder1
  p.setJointMotorControl2(human, left_knee, p.POSITION_CONTROL, targetPosition=left_knee_joint_angles[10] - 1 )  # left_shoulder1


  p.setJointMotorControl2(human, right_hip1, p.POSITION_CONTROL, targetPosition=right_knee_joint_angles[3])  # left_shoulder1
  p.setJointMotorControl2(human, right_hip2, p.POSITION_CONTROL, targetPosition=right_knee_joint_angles[4])  # left_shoulder1
  p.setJointMotorControl2(human, right_hip3, p.POSITION_CONTROL, targetPosition=right_knee_joint_angles[5])  # left_shoulder1
  p.setJointMotorControl2(human, right_knee, p.POSITION_CONTROL, targetPosition=right_knee_joint_angles[6] - 1)  # left_shoulder1



  p.performCollisionDetection()
  points_collision = np.array(p.getContactPoints(human, box_left_knee), dtype=object)

  if(points_collision.size != 0):
    is_hit_left_knee = True
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


# p.stopStateLogging(logid)