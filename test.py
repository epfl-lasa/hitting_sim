import pybullet as p
import json
import time
import pybullet_data


useGUI = True
if useGUI:
  p.connect(p.GUI)
else:
  p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

useZUp = False
useYUp = not useZUp
showJointMotorTorques = False

if useYUp:
  p.configureDebugVisualizer(p.COV_ENABLE_Y_AXIS_UP, 1)


p.resetDebugVisualizerCamera(cameraDistance=7.4,
                             cameraYaw=-94,
                             cameraPitch=-14,
                             cameraTargetPosition=[0.24, -0.02, -0.09])

import pybullet_data
p.setTimeOut(10000)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

timeStep = 1. / 600.

p.setPhysicsEngineParameter(fixedTimeStep=timeStep)

path = pybullet_data.getDataPath() + "/data/motions/humanoid3d_backflip.txt"


#p.loadURDF("plane.urdf",[0,0,-1.03])
print("path	=	", path)
with open(path, 'r') as f:
  motion_dict = json.load(f)
#print("motion_dict	=	", motion_dict)
print("len motion=", len(motion_dict))
print(motion_dict['Loop'])
numFrames = len(motion_dict['Frames'])
print("#frames = ", numFrames)

frameId = p.addUserDebugParameter("frame", 0, numFrames - 1, 0)

erpId = p.addUserDebugParameter("erp", 0, 1, 0.2)

kpMotorId = p.addUserDebugParameter("kpMotor", 0, 1, .2)
forceMotorId = p.addUserDebugParameter("forceMotor", 0, 2000, 1000)

jointTypes = [
    "JOINT_REVOLUTE", "JOINT_PRISMATIC", "JOINT_SPHERICAL", "JOINT_PLANAR", "JOINT_FIXED"
]

startLocations = [[0, 0, 2], [0, 0, 0], [0, 0, -2], [0, 0, -4], [0, 0, 4]]


p.addUserDebugText("Kinematic",
                   [startLocations[3][0], startLocations[3][1] + 1, startLocations[3][2]],
                   [0, 0, 0])

flags=p.URDF_MAINTAIN_LINK_ORDER+p.URDF_USE_SELF_COLLISION

humanoid = p.loadURDF("humanoid/humanoid.urdf",
                       startLocations[3],
                       globalScaling=0.25,
                       useFixedBase=False,
                       flags=flags)



humanoid3_fix = p.createConstraint(humanoid, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                   startLocations[3], [0, 0, 0, 1])

startPose = [
    2, 0.847532, 0, 0.9986781045, 0.01410400148, -0.0006980000731, -0.04942300517, 0.9988133229,
    0.009485003066, -0.04756001538, -0.004475001447, 1, 0, 0, 0, 0.9649395871, 0.02436898957,
    -0.05755497537, 0.2549218909, -0.249116, 0.9993661511, 0.009952001505, 0.03265400494,
    0.01009800153, 0.9854981188, -0.06440700776, 0.09324301124, -0.1262970152, 0.170571,
    0.9927545808, -0.02090099117, 0.08882396249, -0.07817796699, -0.391532, 0.9828788495,
    0.1013909845, -0.05515999155, 0.143618978, 0.9659421276, 0.1884590249, -0.1422460188,
    0.105854014, 0.581348
]


p.resetBasePositionAndOrientation(humanoid, startLocations[3], [0, 0, 0, 1])

index0 = 7

for j in range(p.getNumJoints(humanoid)):
  ji = p.getJointInfo(humanoid, j)
  targetPosition = [0]
  jointType = ji[2]
  if (jointType == p.JOINT_SPHERICAL):
    targetPosition = [0, 0, 0, 1]
    p.setJointMotorControlMultiDof(humanoid,
                                   j,
                                   p.POSITION_CONTROL,
                                   targetPosition,
                                   targetVelocity=[0, 0, 0],
                                   positionGain=0,
                                   velocityGain=1,
                                   force=[1, 1, 1])

  if (jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE):
    p.setJointMotorControl2(humanoid, j, p.VELOCITY_CONTROL, targetVelocity=0, force=10)

  #print(ji)
  print("joint[", j, "].type=", jointTypes[ji[2]])
  print("joint[", j, "].name=", ji[1])

jointIds = []
paramIds = []



chest = 1
neck = 2
rightHip = 3
rightKnee = 4
rightAnkle = 5
rightShoulder = 6
rightElbow = 7
leftHip = 9
leftKnee = 10
leftAnkle = 11
leftShoulder = 12
leftElbow = 13

#rightShoulder=3
#rightElbow=4
#leftShoulder=6
#leftElbow = 7
#rightHip	=	9
#rightKnee=10
#rightAnkle=11
#leftHip = 12
#leftKnee=13
#leftAnkle=14

import time

once = True
p.getCameraImage(320, 200)

while (p.isConnected()):

  if useGUI:
    erp = p.readUserDebugParameter(erpId)
    kpMotor = p.readUserDebugParameter(kpMotorId)
    maxForce = p.readUserDebugParameter(forceMotorId)
    frameReal = p.readUserDebugParameter(frameId)
  else:
    erp = 0.2
    kpMotor = 0.2
    maxForce = 1000
    frameReal = 0

  kp = kpMotor

  frame = int(frameReal)
  frameNext = frame + 1
  if (frameNext >= numFrames):
    frameNext = frame

  frameFraction = frameReal - frame
  #print("frameFraction=",frameFraction)
  #print("frame=",frame)
  #print("frameNext=", frameNext)

  #getQuaternionSlerp

  frameData = motion_dict['Frames'][frame]
  frameDataNext = motion_dict['Frames'][frameNext]

  #print("duration=",frameData[0])
  #print(pos=[frameData])

  basePos1Start = [frameData[1], frameData[2], frameData[3]]
  basePos1End = [frameDataNext[1], frameDataNext[2], frameDataNext[3]]
  basePos1 = [
      basePos1Start[0] + frameFraction * (basePos1End[0] - basePos1Start[0]),
      basePos1Start[1] + frameFraction * (basePos1End[1] - basePos1Start[1]),
      basePos1Start[2] + frameFraction * (basePos1End[2] - basePos1Start[2])
  ]
  baseOrn1Start = [frameData[5], frameData[6], frameData[7], frameData[4]]
  baseOrn1Next = [frameDataNext[5], frameDataNext[6], frameDataNext[7], frameDataNext[4]]
  baseOrn1 = p.getQuaternionSlerp(baseOrn1Start, baseOrn1Next, frameFraction)
  #pre-rotate	to make	z-up


  chestRotStart = [frameData[9], frameData[10], frameData[11], frameData[8]]
  chestRotEnd = [frameDataNext[9], frameDataNext[10], frameDataNext[11], frameDataNext[8]]
  chestRot = p.getQuaternionSlerp(chestRotStart, chestRotEnd, frameFraction)

  neckRotStart = [frameData[13], frameData[14], frameData[15], frameData[12]]
  neckRotEnd = [frameDataNext[13], frameDataNext[14], frameDataNext[15], frameDataNext[12]]
  neckRot = p.getQuaternionSlerp(neckRotStart, neckRotEnd, frameFraction)

  rightHipRotStart = [frameData[17], frameData[18], frameData[19], frameData[16]]
  rightHipRotEnd = [frameDataNext[17], frameDataNext[18], frameDataNext[19], frameDataNext[16]]
  rightHipRot = p.getQuaternionSlerp(rightHipRotStart, rightHipRotEnd, frameFraction)

  rightKneeRotStart = [frameData[20]]
  rightKneeRotEnd = [frameDataNext[20]]
  rightKneeRot = [
      rightKneeRotStart[0] + frameFraction * (rightKneeRotEnd[0] - rightKneeRotStart[0])
  ]

  rightAnkleRotStart = [frameData[22], frameData[23], frameData[24], frameData[21]]
  rightAnkleRotEnd = [frameDataNext[22], frameDataNext[23], frameDataNext[24], frameDataNext[21]]
  rightAnkleRot = p.getQuaternionSlerp(rightAnkleRotStart, rightAnkleRotEnd, frameFraction)

  rightShoulderRotStart = [frameData[26], frameData[27], frameData[28], frameData[25]]
  rightShoulderRotEnd = [
      frameDataNext[26], frameDataNext[27], frameDataNext[28], frameDataNext[25]
  ]
  rightShoulderRot = p.getQuaternionSlerp(rightShoulderRotStart, rightShoulderRotEnd,
                                          frameFraction)

  rightElbowRotStart = [frameData[29]]
  rightElbowRotEnd = [frameDataNext[29]]
  rightElbowRot = [
      rightElbowRotStart[0] + frameFraction * (rightElbowRotEnd[0] - rightElbowRotStart[0])
  ]

  leftHipRotStart = [frameData[31], frameData[32], frameData[33], frameData[30]]
  leftHipRotEnd = [frameDataNext[31], frameDataNext[32], frameDataNext[33], frameDataNext[30]]
  leftHipRot = p.getQuaternionSlerp(leftHipRotStart, leftHipRotEnd, frameFraction)

  leftKneeRotStart = [frameData[34]]
  leftKneeRotEnd = [frameDataNext[34]]
  leftKneeRot = [leftKneeRotStart[0] + frameFraction * (leftKneeRotEnd[0] - leftKneeRotStart[0])]

  leftAnkleRotStart = [frameData[36], frameData[37], frameData[38], frameData[35]]
  leftAnkleRotEnd = [frameDataNext[36], frameDataNext[37], frameDataNext[38], frameDataNext[35]]
  leftAnkleRot = p.getQuaternionSlerp(leftAnkleRotStart, leftAnkleRotEnd, frameFraction)

  leftShoulderRotStart = [frameData[40], frameData[41], frameData[42], frameData[39]]
  leftShoulderRotEnd = [frameDataNext[40], frameDataNext[41], frameDataNext[42], frameDataNext[39]]
  leftShoulderRot = p.getQuaternionSlerp(leftShoulderRotStart, leftShoulderRotEnd, frameFraction)
  leftElbowRotStart = [frameData[43]]
  leftElbowRotEnd = [frameDataNext[43]]
  leftElbowRot = [
      leftElbowRotStart[0] + frameFraction * (leftElbowRotEnd[0] - leftElbowRotStart[0])
  ]

  p.setGravity(0, 0, -10)

  kinematichumanoid = True
  if (kinematichumanoid):
    p.resetJointStateMultiDof(humanoid, chest, chestRot)
    p.resetJointStateMultiDof(humanoid, neck, neckRot)
    p.resetJointStateMultiDof(humanoid, rightHip, rightHipRot)
    p.resetJointStateMultiDof(humanoid, rightKnee, rightKneeRot)
    p.resetJointStateMultiDof(humanoid, rightAnkle, rightAnkleRot)
    p.resetJointStateMultiDof(humanoid, rightShoulder, rightShoulderRot)
    p.resetJointStateMultiDof(humanoid, rightElbow, rightElbowRot)
    p.resetJointStateMultiDof(humanoid, leftHip, leftHipRot)
    p.resetJointStateMultiDof(humanoid, leftKnee, leftKneeRot)
    p.resetJointStateMultiDof(humanoid, leftAnkle, leftAnkleRot)
    p.resetJointStateMultiDof(humanoid, leftShoulder, leftShoulderRot)
    p.resetJointStateMultiDof(humanoid, leftElbow, leftElbowRot)
  p.stepSimulation()

  time.sleep(timeStep)