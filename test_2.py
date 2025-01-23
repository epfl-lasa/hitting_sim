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
    '''
    '''
    obj_virtual = X_obj + np.dot((X - X_obj), v_hit) * v_hit / np.square(np.linalg.norm(v_hit))
    sigma = 0.1
    alpha = np.exp(-np.linalg.norm(X - obj_virtual)/np.square(sigma))
    dX = alpha * v_hit + (1 - alpha) * A1 @ (X - obj_virtual)
    dX = (p_des/lambda_current)*(lambda_current+m_obj) * dX / np.linalg.norm(dX)
    return dX

########################################################################


cid = p.connect(p.SHARED_MEMORY)
if (cid < 0):
  cid = p.connect(p.GUI)
p.resetSimulation()
useRealTime = 1
p.setTimeStep(0.001)
p.setRealTimeSimulation(useRealTime)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)


obUids = p.loadMJCF("mjcf/humanoid_fixed.xml")
human = obUids[0]


box_1 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, 0.3, 0.2], globalScaling=1.0, useFixedBase=0)
box_2 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, -0.3, 0.2], globalScaling=1.0, useFixedBase=0)
box_3 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, 0.3, 1.0], globalScaling=1.0, useFixedBase=0)
box_4 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, -0.3, 1.0], globalScaling=1.0, useFixedBase=0)
p.changeDynamics(box_1, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_2, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_3, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(box_4, -1, mass=0.5, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)                            


table_1 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.7, 0.4, -0.2], globalScaling=1.0, useFixedBase=1)
table_2 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.7, -0.4, -0.2], globalScaling=1.0, useFixedBase=1)
table_3 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.7, 0.4, 0.6], globalScaling=1.0, useFixedBase=1)
table_4 = p.loadURDF("descriptions/robot_descriptions/objects_description/objects/table_plane.urdf",
                                [0.7, -0.4, 0.6], globalScaling=1.0, useFixedBase=1)
p.changeDynamics(table_1, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(table_2, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(table_3, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)
p.changeDynamics(table_4, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)

plane = p.loadURDF("plane_transparent.urdf")
p.changeDynamics(plane, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)

# Print joint and link information

print("Joint and Link Mapping:")
for i in range(p.getNumJoints(human)):
    joint_info = p.getJointInfo(human, i)
    joint_index = joint_info[0]
    joint_name = joint_info[1].decode('utf-8')  # Joint name
    link_name = joint_info[12].decode('utf-8')  # Child link name
    print(f"Joint Index: {joint_index}, Joint Name: {joint_name}, Link Name: {link_name}")

# Print base/root link information
print("\nBase Link:")


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

start_position = [-0.1, 0.3, 1.0]
target_position = [1.0, 0.3, 1.0]
p.addUserDebugPoints([start_position], [[1.0, 0.0, 0.0]], 50, 0)


# print("somthinggg  ", p.calculateInverseKinematics(human, 11, [0.5, 0.3, 0.2], [0, 0, 0, 1]))

joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=31,  # Link index for left_hand
    targetPosition=start_position,
    # lowerLimits=[-3.14, -3.14, -1.57],  # Joint limits for shoulder and elbow
    # upperLimits=[3.14, 3.14, 2.0],
    # jointRanges=[6.28, 6.28, 3.57],  # Allowable joint ranges
    # restPoses=[0, 0, 0],  # Resting positions
    # jointDamping=[0.1, 0.1, 0.1],  # Damping for stability
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

print("Joint angles: ", joint_angles)

init = 0


p.setJointMotorControl2(human, 26, p.POSITION_CONTROL, targetPosition=joint_angles[14])  # left_shoulder1
p.setJointMotorControl2(human, 27, p.POSITION_CONTROL, targetPosition=joint_angles[15])  # left_shoulder1
p.setJointMotorControl2(human, 29, p.POSITION_CONTROL, targetPosition=joint_angles[16])  # left_shoulder1

p.stepSimulation()

time.sleep(5)


###########################################

A = np.array([[-10, 0, 0], [0, -2, 0], [0, 0, -2]])

while (1):

  left_hand_position = p.getLinkState(human, 31)[4]
  des_vel = linear_ds(A, np.array(left_hand_position), np.array(target_position))

  next_position = np.array(left_hand_position) + des_vel*0.001

  joint_angles = p.calculateInverseKinematics(
    bodyUniqueId=human,
    endEffectorLinkIndex=31,  # Link index for left_hand
    targetPosition=next_position,
    # lowerLimits=[-3.14, -3.14, -1.57],  # Joint limits for shoulder and elbow
    # upperLimits=[3.14, 3.14, 2.0],
    # jointRanges=[6.28, 6.28, 3.57],  # Allowable joint ranges
    # restPoses=[0, 0, 0],  # Resting positions
    # jointDamping=[0.1, 0.1, 0.1],  # Damping for stability
    solver=p.IK_DLS  # Use Damped Least Squares solver
  )

  p.setJointMotorControl2(human, 26, p.POSITION_CONTROL, targetPosition=joint_angles[14])  # left_shoulder1 
  p.setJointMotorControl2(human, 27, p.POSITION_CONTROL, targetPosition=joint_angles[15])  # left_shoulder1
  p.setJointMotorControl2(human, 29, p.POSITION_CONTROL, targetPosition=joint_angles[16])  # left_shoulder1


  # time.sleep(0.01)
  # kneeAngleTarget = p.readUserDebugParameter(kneeAngleTargetId)
  # maxForce = p.readUserDebugParameter(maxForceId)
  # p.setJointMotorControl2(human,
  #                         kneeJointIndex,
  #                         p.POSITION_CONTROL,
  #                         targetPosition=kneeAngleTarget,
  #                         force=maxForce)

  # hipAngleLeftXTarget = p.readUserDebugParameter(HipTargetLeftIdX)
  # maxForce = p.readUserDebugParameter(HipMaxForceLeftId)
  # p.setJointMotorControl2(human,
  #                         hipIndexLeftX,
  #                         p.POSITION_CONTROL,
  #                         targetPosition=hipAngleLeftXTarget,
  #                         force=maxForce)

  # kneeAngleTargetLeft = p.readUserDebugParameter(kneeAngleTargetLeftId)
  # maxForceLeft = p.readUserDebugParameter(maxForceLeftId)
  # p.setJointMotorControl2(human,
  #                         kneeJointIndexLeft,
  #                         p.POSITION_CONTROL,
  #                         targetPosition=kneeAngleTargetLeft,
  #                         force=maxForceLeft)
  # q = 0.9
  
  # p.addUserDebugPoints([p.getLinkState(human, 31)[4]], [[1.0, 0.0, 0.0]], 50, 0)
  # q = q + 0.1

  p.stepSimulation()