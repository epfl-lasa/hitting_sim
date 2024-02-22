import pybullet as p
import pybullet_data
import numpy as np
import math
from functions import get_stein_divergence
import pybullet_utils.bullet_client as bc

class sim_robot_env:
    def __init__(self, use_sim, box_object, counter):

        if counter == 0:
            mode = p.DIRECT
        else:
            mode = p.GUI

        self.physicsClient = bc.BulletClient(mode
                                              , options='--background_color_red=0 --background_color_green=0' +
                             ' --background_color_blue=0 --width=1000 --height=1000')
        self.physicsClient.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physicsClient.resetSimulation()
        self.plane = self.physicsClient.loadURDF("plane_transparent.urdf")
        self.physicsClient.changeVisualShape(self.plane, -1, rgbaColor=[0, 0, 0, 0])
        startPos = [0, 0, 0]
        startOrientation = self.physicsClient.getQuaternionFromEuler([0, 0, 0])
        self.physicsClient.setGravity(0, 0, -9.81)
        self.physicsClient.setTimeStep(0.001)
        self.physicsClient.setRealTimeSimulation(use_sim)
        self.physicsClient.resetDebugVisualizerCamera(cameraDistance=1.60, cameraYaw=200, cameraPitch=-25.00,
                                            cameraTargetPosition=[0, 0, 0])
        
        self.physicsClientID = self.physicsClient._client

        self.robot = p.loadURDF("kuka_iiwa/model.urdf", startPos, startOrientation, useFixedBase=1)
        # self.robot = self.physicsClient.loadURDF("urdfs/franka_panda/panda.urdf", startPos, startOrientation, useFixedBase=True)

        '''
        Create a scene too
        the environment with the box and table for the different scenarios I would like in the simulation
        '''
        if counter == 1:
            self.box = self.physicsClient.loadURDF("descriptions/robot_descriptions/objects_description/objects/simple_box.urdf",
                                [0.5, 0.3, 0.2], globalScaling=1.0, useFixedBase=0)
            # tableOrientation = self.physicsClient.getQuaternionFromEuler([0, 0, math.pi / 2])
            # self.table = self.physicsClient.loadURDF("descriptions/robot_descriptions/objects_description/objects/table.urdf",
                            # [1.15, 0.45, 0.0], tableOrientation, globalScaling=1.0, useFixedBase=1)
            self.physicsClient.changeDynamics(self.box, -1, mass=box_object.mass, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                            spinningFriction=0.02, restitution=0, lateralFriction=0.3)
            self.physicsClient.changeDynamics(self.plane, -1, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
                         spinningFriction=0.02, restitution=0, lateralFriction=0.3)
            # self.physicsClient.changeDynamics(self.table, 1, mass=10, linearDamping=0.04, angularDamping=0.04, rollingFriction=0.01,
            #                 spinningFriction=0.02, restitution=0, lateralFriction=0.15)

        
        self.numJoints = self.physicsClient.getNumJoints(self.robot)

        self.ee_id = 6
        self.zeros = [0.0] * 7
        self.relative_ee = [0, 0, 0]

        # iiwa
        self.q_dot_ul = np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])
        self.q_dot_ll = -np.array([1.48, 1.48, 1.74, 1.30, 2.26, 2.35, 2.35])

        self.q_ll = -np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])
        self.q_ul = np.array([2.96, 2.09, 2.96, 2.09, 2.96, 2.09, 3.05])

        # self.rest_pose = np.array([-0.4, 0.8, -0.1, -1.6, 0.0, 0.4, 0.0])
        self.rest_pose = np.array([-0.6, 0.8, 0.3, -1.6, 1.0, 1.75, 0.0]) # Good position for hitting
        # self.rest_pose = np.array([-0.6, 0.8, 0.3, -1.6, 1.0, -2.0, 0.0]) # Good position for hitting for other joints


        # panda
        # self.q_dot_ul = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])
        # self.q_dot_ll = -np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61])

        # self.q_ul = np.array([2.89, 1.76, 2.89, -0.06, 2.89, 3.75, 2.89])
        # self.q_ll = -np.array([2.89, 1.76, 2.89, 3.07, 2.89, 0.01, 2.89])

        # self.rest_pose = self.q_ll + (self.q_ul - self.q_ll) / 2 + np.array([-0.9, 0, 0, 0, math.pi/2, 0.0, 0])

    def step(self):
        self.physicsClient.stepSimulation()

    def get_IK_joint_position(self, x):
        return self.physicsClient.calculateInverseKinematics(self.robot, self.ee_id, x, restPoses=self.rest_pose, lowerLimits=self.q_ll, upperLimits=self.q_ul)
    
    def get_IK_joint_position_orientation(self, x, orientation):
        return self.physicsClient.calculateInverseKinematics(self.robot, self.ee_id, targetPosition=x, targetOrientation=orientation, restPoses=self.rest_pose, lowerLimits=self.q_ll, upperLimits=self.q_ul)

    def get_IK_joint_position_point(self, x, point_id):
        # return self.physicsClient.calculateInverseKinematics(self.robot, point_id, x, restPoses=self.rest_pose, lowerLimits=self.q_ll, upperLimits=self.q_ul)
        return self.physicsClient.calculateInverseKinematics(self.robot, point_id, x, lowerLimits=self.q_ll, upperLimits=self.q_ul)


    def set_to_joint_position(self, q):
        for i in range(self.numJoints):
            self.physicsClient.resetJointState(self.robot, i, q[i])

    def move_with_joint_velocities(self, q_dot):
        self.physicsClient.stepSimulation()
        self.physicsClient.setJointMotorControlArray(self.robot, range(self.numJoints), controlMode=self.physicsClient.VELOCITY_CONTROL,
                               targetVelocities=q_dot.tolist())

    def move_to_joint_position(self, q, q_dot):
        self.physicsClient.stepSimulation()
        for i in range(self.numJoints):
            self.physicsClient.setJointMotorControl2(self.robot, jointIndex=i, controlMode=self.physicsClient.POSITION_CONTROL,
                                    targetPosition=q[i],
                                    targetVelocity=q_dot[i],
                                    force=500,
                                    positionGain=1,
                                    velocityGain=1)

    def get_joint_position(self):
        joint_state = self.physicsClient.getJointStates(self.robot, range(self.numJoints))
        joint_position = [state[0] for state in joint_state]
        return joint_position

    def get_ee_position(self):
        return self.physicsClient.getLinkState(self.robot, self.ee_id)[0]
    
    def get_point_position(self, point_id):
        return self.physicsClient.getLinkState(self.robot, point_id)[4]
    
    def get_joint_cartesian_position(self, joint_id):
        return self.physicsClient.getLinkState(self.robot, joint_id)[4]
    
    def get_relative_link_com_position(self, link_id):
        return self.physicsClient.getLinkState(self.robot, link_id)[2]
    
    def get_multi_link_position(self, link_ids):
        pos = []
        all_pos = self.physicsClient.getLinkStates(self.robot, link_ids)
        for i in range(len(link_ids)):
            pos.append(all_pos[i][0])
        return pos

    def get_multi_joint_position(self, joint_ids):
        pos = []
        all_pos = self.physicsClient.getLinkStates(self.robot, joint_ids)
        for i in range(len(joint_ids)):
            pos.append(all_pos[i][4])
        return pos
    '''
    Multiple different jacobian functions
    '''


    def get_trans_jacobian(self):
        q = self.get_joint_position()
        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian(self):
        q = self.get_joint_position()
        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q, self.zeros, self.zeros)
        return jac_r_fn
    
    def get_trans_jacobian_point(self, point_id):
        q = self.get_joint_position()
        relative_dist = 1* np.array(self.get_relative_link_com_position(point_id))
        relative_dist = relative_dist.tolist()
        # print("relative dist ", relative_dist)
        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, point_id, relative_dist, q, self.zeros, self.zeros)
        # jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, point_id, self.relative_ee, q, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian_point(self, point_id):
        q = self.get_joint_position()
        relative_dist = -1* self.get_relative_link_com_position(point_id)

        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, point_id, relative_dist, q, self.zeros, self.zeros)
        return jac_r_fn

    def get_trans_jacobian_specific(self, q_specific):
        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian_specific(self, q_specific):
        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, self.ee_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        return jac_r_fn
    
    def get_trans_jacobian_specific_point(self, q_specific, point_id):
        relative_dist = self.get_relative_link_com_position(point_id)
        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, point_id, relative_dist, q_specific, self.zeros, self.zeros)
        return jac_t_fn

    def get_rot_jacobian_specific_point(self, q_specific, point_id):
        jac_t_fn, jac_r_fn = self.physicsClient.calculateJacobian(self.robot, point_id, self.relative_ee, q_specific, self.zeros, self.zeros)
        return jac_r_fn
    
    '''
    Multiple different mass matrix functions
    '''

    def get_mass_matrix_specific(self, q):
        return self.physicsClient.calculateMassMatrix(self.robot, q)

    def get_mass_matrix(self):
        q = self.get_joint_position()
        return self.physicsClient.calculateMassMatrix(self.robot, q)

    def get_inertia_matrix(self):
        J = self.get_trans_jacobian()
        M = self.get_mass_matrix()
        return np.array(np.linalg.inv(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T))
    
    def get_inv_inertia_matrix(self):
        J = self.get_trans_jacobian()
        M = self.get_mass_matrix()
        return np.array(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T)

    def get_effective_inertia(self, direction):
        inv_inertia = self.get_inv_inertia_matrix()
        return 1/(direction.T @ inv_inertia @ direction)

    def get_inverse_effective_inertia(self, direction):
        inv_inertia = self.get_inv_inertia_matrix()
        return (direction.T @ inv_inertia @ direction)




    def get_inertia_matrix_specific(self, q):
        J = self.get_trans_jacobian_specific(q)
        M = self.get_mass_matrix_specific(q)
        return np.array(np.linalg.inv(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T))

    def get_inv_inertia_matrix_specific(self, q):
        J = self.get_trans_jacobian_specific(q)
        M = self.get_mass_matrix_specific(q)
        return np.array(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T)

    def get_inverse_effective_inertia_specific(self, q, direction):
        inv_inertia = self.get_inv_inertia_matrix_specific(q)
        return (direction.T @ inv_inertia @ direction)
    
    def get_effective_inertia_specific(self, q, direction):
        inv_inertia = self.get_inv_inertia_matrix_specific(q)
        return 1/(direction.T @ inv_inertia @ direction)
    





    def get_inertia_matrix_point(self, point_id):
        J = self.get_trans_jacobian_point(point_id)
        M = self.get_mass_matrix()
        return np.array(np.linalg.inv(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T))
    
    def get_inv_inertia_matrix_point(self, point_id):
        J = self.get_trans_jacobian_point(point_id)
        M = self.get_mass_matrix()
        return np.array(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T)

    def get_effective_inertia_point(self, direction, point_id):
        inv_inertia = self.get_inv_inertia_matrix_point(point_id)
        return 1/(direction.T @ inv_inertia @ direction)

    def get_inverse_effective_inertia_point(self, direction, point_id):
        inv_inertia = self.get_inv_inertia_matrix_point(point_id)
        return (direction.T @ inv_inertia @ direction)






    def get_inertia_matrix_specific_point(self, q, point_id):
        J = self.get_trans_jacobian_specific_point(q, point_id)
        M = self.get_mass_matrix_specific(q)
        return np.array(np.linalg.inv(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T))

    def get_inv_inertia_matrix_specific_point(self, q, point_id):
        J = self.get_trans_jacobian_specific_point(q, point_id)
        M = self.get_mass_matrix_specific(q)
        return np.array(np.array(J) @ np.linalg.inv(np.array(M)) @ np.array(J).T)
    
    def get_effective_inertia_specific_point(self, q, direction, point_id):
        inv_inertia = self.get_inv_inertia_matrix_specific_point(q, point_id)
        return 1/(direction.T @ inv_inertia @ direction)

    def get_inverse_effective_inertia_specific_point(self, q, direction, point_id):
        inv_inertia = self.get_inv_inertia_matrix_specific_point(q, point_id)
        return (direction.T @ inv_inertia @ direction)



    '''
    Stein divergence functions
    '''

    def get_stein_divergence_joint_gradient(self, g_current, A_d):
        num_joints = self.numJoints
        dg_dq = np.zeros((num_joints, 1))
        dq = 0.01
        q_current = self.get_joint_position()
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            jac_t_fn = self.get_trans_jacobian_specific(q_new)
            mass_fn = self.physicsClient.calculateMassMatrix(self.robot, q_new)
            A_new = np.array(np.linalg.inv(np.array(jac_t_fn) @ np.linalg.inv(np.array(mass_fn)) @ np.array(jac_t_fn).T))
            g_new = get_stein_divergence(A_new, A_d)
            dg_dq[l] = ((g_new - g_current) / dq)
        return dg_dq


    '''
    Functions to get the velocity of the end effector
    '''
    def get_ee_velocity_current(self):
        return self.physicsClient.getLinkState(self.robot, self.ee_id, computeLinkVelocity=True)[6]
    
    def get_velocity_point_current(self, point_id):
        return self.physicsClient.getLinkState(self.robot, point_id, computeLinkVelocity=True)[6]


    '''
    Gradient functions
    '''
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
    
    def get_effective_inertia_gradient(self, direction):
        num_joints = self.numJoints
        dL_dq_dir = np.zeros((num_joints, 1))
        dq = 0.001
        q_current = self.get_joint_position()
        eff_lambda_current = self.get_effective_inertia_specific(q_current, direction)
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            eff_lambda_new = self.get_effective_inertia_specific(q_new, direction)
            dL_dq_dir[l] = (eff_lambda_new - eff_lambda_current) / dq
        return dL_dq_dir
    
    def get_effective_inertia_point_gradient(self, direction, point_id):
        num_joints = self.numJoints
        dL_dq_dir = np.zeros((num_joints, 1))
        dq = 0.001
        q_current = self.get_joint_position()
        eff_lambda_current = self.get_effective_inertia_specific_point(q_current, direction, point_id)
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            eff_lambda_new = self.get_effective_inertia_specific_point(q_new, direction, point_id)
            dL_dq_dir[l] = (eff_lambda_new - eff_lambda_current) / dq
        return dL_dq_dir
    
    def get_inverse_effective_inertia_gradient(self, direction):
        num_joints = self.numJoints
        dL_dq_dir = np.zeros((num_joints, 1))
        dq = 0.001
        q_current = self.get_joint_position()
        eff_lambda_current = self.get_inverse_effective_inertia_specific(q_current, direction)
        for l in range(num_joints):
            q_new = q_current.copy()
            q_new[l] = q_new[l] + dq
            eff_lambda_new = self.get_inverse_effective_inertia_specific(q_new, direction)
            dL_dq_dir[l] = (eff_lambda_new - eff_lambda_current) / dq
        return dL_dq_dir
    
    def get_effective_inertia_influence_matrix(self, direction):
        dL_dq_dir = self.get_effective_inertia_gradient(direction)
        dL_dq_dir[dL_dq_dir < 0] = 0
        # dL_dq_dir[dL_dq_dir > 0] = 1
        return np.array(np.diag(dL_dq_dir.flatten()))

    def get_effective_inertia_point_influence_matrix(self, direction, point_id):
        dL_dq_dir = self.get_effective_inertia_point_gradient(direction, point_id)
        # dL_dq_dir[dL_dq_dir < 0] = 0
        # # dL_dq_dir[dL_dq_dir > 0] = 1
        return np.array(np.diag(dL_dq_dir.flatten()))
    
    def get_velocity_manipulability_metric(self):
        jac = self.get_trans_jacobian()
        m = np.sqrt(np.linalg.det(np.array(jac) @ np.array(jac).T))
        return m

    def reset_box(self, box_position, box_orientation):
        self.physicsClient.resetBasePositionAndOrientation(self.box, box_position, box_orientation)
        return 0

    def get_box_position_orientation(self):
        return self.physicsClient.getBasePositionAndOrientation(self.box)

    def get_box_velocity(self):
        return self.physicsClient.getBaseVelocity(self.box)[0]

    def get_box_speed(self):
        return np.linalg.norm(np.array(self.physicsClient.getBaseVelocity(self.box)[0]))

    def get_collision_points(self):
         p.performCollisionDetection(self.physicsClientID)
         return np.array(self.physicsClient.getContactPoints(self.robot, self.box), dtype=object)

    def get_collision_position(self):
        collision_points = self.get_collision_points()
        return collision_points[0][5]

    def get_mesh_vertices(self):
        return self.physicsClient.getMeshData(self.box)
    

############################## DEBUG LINES ##############################################

    def draw_line(self, point1, point2, color, linewidth, trailduration):
        self.physicsClient.addUserDebugLine(point1, point2, color, linewidth, trailduration)

    def draw_point(self, point, color, pointsize, trailduration):
        self.physicsClient.addUserDebugPoints(point, color, pointsize, trailduration)

    
