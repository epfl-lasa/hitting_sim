import pybullet as p
import pybullet_data
import numpy as np
import math
from functions import get_stein_divergence

class sim_robot:
    def __init__(self, use_sim):

        self.physicsClient = p.connect(p.GUI, options='--background_color_red=1 --background_color_green=1' +
                             ' --background_color_blue=1 --width=1000 --height=1000')
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        plane = p.loadURDF("plane_transparent.urdf")
        p.changeVisualShape(plane, -1, rgbaColor=[0, 0, 0, 0])
        startPos = [0, 0, 0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.001)
        p.setRealTimeSimulation(use_sim)
        p.resetDebugVisualizerCamera(cameraDistance=1.60, cameraYaw=200, cameraPitch=-25.00,
                                            cameraTargetPosition=[0, 0, 0])

        # self.robot = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation, useFixedBase=1)
        self.robot = p.loadURDF("urdfs/franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=True)

        '''
        Create a scene too
        the environment with the box and table for the different scenarios I would like in the simulation
        '''
        
        self.numJoints = p.getNumJoints(self.robot)
        self.ee_id = 5
        self.zeros = [0.0] * 7
        self.relative_ee = [0, 0, 0]

        # iiwa
        # self.q_dot_ul = np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])
        # self.q_dot_ll = -np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])

        # self.q_ll = -np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
        # self.q_ul = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])


        # panda
        self.q_dot_ul = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        self.q_dot_ll = -np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

        self.q_ul = np.array([2.89, 1.76, 2.89, -0.06, 2.89, 3.75, 2.89])
        self.q_ll = -np.array([2.89, 1.76, 2.89, 3.07, 2.89, 0.01, 2.89])

        self.rest_pose = self.q_ll + (self.q_ul - self.q_ll) / 2 + np.array([-0.8, 0, 0, 0, math.pi/2, 0.0, 0])

    def step(self):
        p.stepSimulation()

    def get_IK_joint_position(self, x):
        return p.calculateInverseKinematics(self.robot, self.ee_id, x, restPoses=self.rest_pose, lowerLimits=self.q_ll, upperLimits=self.q_ul)

    def set_to_joint_position(self, q):
        for i in range(self.numJoints):
            p.resetJointState(self.robot, i, q[i])

    def move_with_joint_velocities(self, q_dot):
        p.stepSimulation()
        p.setJointMotorControlArray(self.robot, range(self.numJoints), controlMode=p.VELOCITY_CONTROL,
                               targetVelocities=q_dot.tolist())

    def move_to_joint_position(self, q, q_dot):
        p.stepSimulation()
        for i in range(self.numJoints):
            p.setJointMotorControl2(self.robot, jointIndex=i, controlMode=p.POSITION_CONTROL,
                                    targetPosition=q[i],
                                    targetVelocity=q_dot[i],
                                    force=500,
                                    positionGain=1,
                                    velocityGain=1)

    def get_joint_position(self):
        joint_state = p.getJointStates(self.robot, range(self.numJoints))
        joint_position = [state[0] for state in joint_state]
        return joint_position

    def get_ee_position(self):
        return p.getLinkState(self.robot, self.ee_id)[0]

    def get_trans_jacobian(self):
        q = self.get_joint_position()
        jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian(self):
        q = self.get_joint_position()
        jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        return jac_r_fn

    def get_trans_jacobian_specific(self, q_specific):
        jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian_specific(self, q_specific):
        jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        return jac_r_fn

    def get_mass_matrix_specific(self, q):
        return p.calculateMassMatrix(self.robot, q)

    def get_mass_matrix(self):
        q = self.get_joint_position()
        return p.calculateMassMatrix(self.robot, q)

    def get_inertia_matrix_specific(self, q):
        J = self.get_trans_jacobian_specific(q)
        M = self.get_mass_matrix_specific(q)
        return np.array(np.linalg.inv(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T))

    def get_inv_inertia_matrix_specific(self, q):
        J = self.get_trans_jacobian_specific(q)
        M = self.get_mass_matrix_specific(q)
        return np.array(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T)

    def get_inertia_matrix(self):
        q = self.get_joint_position()
        J = self.get_trans_jacobian()
        M = self.get_mass_matrix()
        return np.array(np.linalg.inv(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T))

    def get_stein_divergence_joint_gradient(self, g_current, A_d):
        num_joints = self.numJoints
        dg_dq = np.zeros((num_joints, 1))
        dq = 0.01
        q_current = self.get_joint_position()
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            jac_t_fn = self.get_trans_jacobian_specific(q_new)
            mass_fn = p.calculateMassMatrix(self.robot, q_new)
            A_new = np.array(np.linalg.inv(np.array(jac_t_fn) @ np.linalg.inv(np.array(mass_fn)) @ np.array(jac_t_fn).T))
            g_new = get_stein_divergence(A_new, A_d)
            dg_dq[l] = ((g_new - g_current) / dq)
        return dg_dq

    def get_ee_velocity_current(self):
        return p.getLinkState(self.robot, self.ee_id, computeLinkVelocity=True)[6]

    def get_directional_inertia_gradient(self, direction):
        num_joints = self.numJoints
        dL_dq_dir = np.zeros((num_joints, 1))
        dq = 0.001
        q_current = self.get_joint_position()
        A_current = self.get_inertia_matrix_specific(q_current)
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            A_new = self.get_inertia_matrix_specific(q_new)
            dL_dq = (A_new - A_current) / dq

            dL_dq_dir[l] = direction.T @ dL_dq @ direction
        return dL_dq_dir

    def get_directional_inertia_gradient_SPD(self, direction):
        return 0

    def get_velocity_manipulability_metric(self):
        jac = self.get_trans_jacobian()
        m = np.sqrt(np.linalg.det(np.array(jac) @ np.array(jac).T))
        return m


         
