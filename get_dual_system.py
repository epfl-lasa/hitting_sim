import pybullet as p
import pybullet_data
import numpy as np

from functions import get_stein_divergence


class sim_dual_robot:
    def __init__(self):

        physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetSimulation()
        plane = p.loadURDF("plane.urdf")
        self.startPos_l = [-0.5, 0, 0]
        self.startPos_f = [ 0.5, 0, 0]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(0.0001)
        p.setRealTimeSimulation(0)

        self.robot_l = p.loadURDF("kuka_iiwa/model.urdf", self.startPos_l, startOrientation, useFixedBase=1)
        self.robot_f = p.loadURDF("kuka_iiwa/model.urdf", self.startPos_f, startOrientation, useFixedBase=1)
        self.numJoints = p.getNumJoints(self.robot_l)
        self.ee_id = 6
        self.zeros = [0.0] * 7
        self.relative_ee = [0, 0, 0]

        self.q_dot_ul = np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])
        self.q_dot_ll = -np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])

    def set_to_joint_position(self, q, which):
        if which == "leader":
            for i in range(self.numJoints):
                p.resetJointState(self.robot_l, i, q[i])
        else:
            for i in range(self.numJoints):
                p.resetJointState(self.robot_f, i, q[i])

    def get_joint_position(self, which):
        if which == "leader":
            joint_state = p.getJointStates(self.robot_l, range(self.numJoints))
        else:
            joint_state = p.getJointStates(self.robot_f, range(self.numJoints))
        joint_position = [state[0] for state in joint_state]
        return joint_position

    def get_ee_position(self, which):
        if which == "leader":
            return p.getLinkState(self.robot_l, self.ee_id)[0]
        else:
            return p.getLinkState(self.robot_f, self.ee_id)[0]

    def get_trans_jacobian(self, which):
        q = self.get_joint_position(which)
        if which == "leader":
            jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot_l, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        else:
            jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot_f, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian(self, which):
        q = self.get_joint_position(which)
        if which == "leader":
            jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot_l, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        else:
            jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot_f, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        return jac_r_fn

    def get_trans_jacobian_specific(self, q_specific, which):
        if which == "leader":
            jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot_l, self.ee_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        else:
            jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot_f, self.ee_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian_specific(self, q_specific):
        jac_t_fn, jac_r_fn = p.calculateJacobian(self.robot_l, self.ee_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        return jac_r_fn

    def get_mass_matrix(self, q):
        jac = self.get_trans_jacobian_specific(q)
        return p.calculateMassMatrix(self.robot_l, q)

    def get_inertia_matrix(self, q):
        J = self.get_trans_jacobian_specific(q)
        M = self.get_mass_matrix(q)
        return np.array(np.linalg.inv(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T))

    def get_stein_divergence_joint_gradient(self, g_current, A_d):
        num_joints = self.numJoints
        dg_dq = np.zeros((num_joints, 1))
        dq = 0.001
        q_current = self.get_joint_position()
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            jac_t_fn = self.get_trans_jacobian_specific(q_new)
            mass_fn = p.calculateMassMatrix(self.robot_l, q_new)
            A_new = np.array(np.linalg.inv(np.array(jac_t_fn) @ np.linalg.inv(np.array(mass_fn)) @ np.array(jac_t_fn).T))
            g_new = get_stein_divergence(A_new, A_d)
            dg_dq[l] = ((g_new - g_current) / dq)
        return dg_dq

    def get_ee_velocity_current(self):
        return p.getLinkState(self.robot_l, self.ee_id, 1)[0]

    def get_directional_inertia_gradient(self, direction):
        num_joints = self.numJoints
        # dL_dq = np.zeros((3, 3, num_joints))
        dL_dq_dir = np.zeros((num_joints, 1))
        dq = 0.001
        q_current = self.get_joint_position()
        A_current = self.get_inertia_matrix(q_current)
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            A_new = self.get_inertia_matrix(q_new)
            dL_dq = (A_new - A_current) / dq

            dL_dq_dir[l] = direction.T @ dL_dq @ direction
        return dL_dq_dir

    def get_directional_inertia_gradient_SPD(self, direction):
        return 0