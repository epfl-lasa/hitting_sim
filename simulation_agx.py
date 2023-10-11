#|
#|    Copyright (C) 2021-2023 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
#|    Authors: Harshit Khurana (maintainer)
#|
#|    email:   harshit.khurana@epfl.ch
#|
#|    Other contributors:
#|             Elise Jeandupeux (elise.jeandupeux@epfl.ch)
#|
#|    website: lasa.epfl.ch
#|
#|    This file is part of iam_dual_arm_control.
#|    This work was supported by the European Community's Horizon 2020 Research and Innovation
#|    programme (call: H2020-ICT-09-2019-2020, RIA), grant agreement 871899 Impact-Aware Manipulation.
#|
#|    iam_dual_arm_control is free software: you can redistribute it and/or modify  it under the terms
#|    of the GNU General Public License as published by  the Free Software Foundation,
#|    either version 3 of the License, or  (at your option) any later version.
#|
#|    iam_dual_arm_control is distributed in the hope that it will be useful,
#|    but WITHOUT ANY WARRANTY; without even the implied warranty of
#|    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#|    GNU General Public License for more details.
#|

import numpy as np
import time
import os

from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS
from get_robot import sim_robot_env
from iiwa_environment import object
import functions as f

from scipy.spatial.transform import Rotation as R
from roboticstoolbox.robot.ERobot import ERobot

# AGX
from pclick import Client
from pclick import MessageFactory

def reset_sim_agx():
    # Send 0 velocity
    message = MessageFactory.create_controlmessage()
    robot_msg = message.objects["robot"]
    robot_msg.angleVelocities.extend([0, 0, 0, 0, 0, 0, 0])
    client.send(message)
    response = client.recv()
    # Reset Sim
    message = MessageFactory.create_resetmessage()
    client.send(message)
    response = client.recv()


def agx_send_vel_command(q_dot):
    message = MessageFactory.create_controlmessage()
    robot_msg = message.objects["robot"]
    robot_msg.angleVelocities.extend(list(q_dot))
    client.send(message)
    response = client.recv()


def update_q_agx_pybullet():
    message = MessageFactory.create_sensorrequestmessage()
    client.send(message)
    response = client.recv()

    robot.q = np.array(response.objects['robot'].angleSensors)
    iiwa.set_to_joint_position(robot.q)


if __name__ == "__main__":
    # Connect to AGX sim
    addr = f"tcp://localhost:5555"
    client = Client()
    print(f"Connecting to click server {addr}")
    client.connect(addr)

    # Reset pos AGX + 0 velocity
    reset_sim_agx()

    # Get Sensor message
    message = MessageFactory.create_sensorrequestmessage()
    client.send(message)
    response = client.recv()

    # Robot init pose
    q_init = np.array(response.objects['robot'].angleSensors)
    dq_init = np.array(response.objects['robot'].angleVelocitySensors)
    torque_init = np.array(response.objects['robot'].torqueSensors)

    # Box init pose
    box_position_init = response.objects['Box'].objectSensors[0].position.arr
    box_ori = response.objects['Box'].objectSensors[1].rpy.arr
    r = R.from_euler('xzy', [box_ori[0], box_ori[1], box_ori[2]], degrees=True)
    box_orientation_init = r.as_quat()

    # Init Pybullet to get inertia
    box = object.Box([0.2, 0.2, 0.2], 0.5)
    iiwa = sim_robot_env(1, box)
    iiwa.set_to_joint_position(q_init)

    ###################### Robot RBDyn ##################

    robot = ERobot.URDF(os.path.dirname(os.path.realpath('__file__')) + "/urdf_models/iiwa-pybullet.urdf")

    ######################### PARAMETERS ###############################
    trailDuration = 0  # Make it 0 if you don't want the trail to end
    contactTime = 0.5  # This is the time that the robot will be in contact with the box

    ###################### INIT CONDITIONS #################################
    X_init = [0.3, -0.2, 0.5]
    robot.q = q_init

    iiwa.set_to_joint_position(q_init)
    Lambda_init = iiwa.get_inertia_matrix_specific(tuple(q_init))

    ##################### DS PROPERTIES ####################################
    A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])
    h_dir = np.array([0, 1, 0])  # This is the direction of the hitting

    X_ref_grid = f.des_hitting_point_grid(box, box_position_init, 0, 5)

    ########################################################################

    for X_ref in X_ref_grid:
        X_ref = box_position_init

        # Reset pos AGX + 0 velocity
        reset_sim_agx()

        robot.q = q_init
        iiwa.set_to_joint_position(q_init)
        Lambda_init = iiwa.get_inertia_matrix_specific(tuple(q_init))

        lambda_dir = h_dir.T @ Lambda_init @ h_dir
        is_hit = False

        # take some time
        time.sleep(1)

        # initialise the time
        time_init = time.time()

        # Start the motion
        box_position_prev = box_position_init
        time_prev = time.time()
        time_sim_prev = 0
        while 1:
            update_q_agx_pybullet()

            X_qp = np.array(robot.fkine(robot.q))[:3, 3]

            if not is_hit:
                dX = linear_hitting_ds_pre_impact(
                    A, X_qp, X_ref, h_dir, 0.7, lambda_dir, box.mass)
            else:
                dX = linear_ds(A, X_qp, X_ref)

            hit_dir = dX / np.linalg.norm(dX)

            lambda_current = iiwa.get_inertia_matrix_specific(tuple(robot.q))
            lambda_dir = hit_dir.T @ lambda_current @ hit_dir
            jac = np.array(robot.jacob0(robot.q))[:3, :]

            q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(
                dX, jac, iiwa, hit_dir, 0.15, lambda_dir)

            agx_send_vel_command(q_dot)

            message = MessageFactory.create_sensorrequestmessage()
            client.send(message)
            response = client.recv()

            # Detect hit - Need something more here later
            if (np.linalg.norm(np.array(response.objects['Box'].objectSensors[0].position.arr) - box_position_init) > 0.01):
                is_hit = True

            time_sim = response.simVars.simulatedTime
            box_vel = (np.array(box_position_prev) -
                       np.array(response.objects['Box'].objectSensors[0].position.arr))/(time_sim_prev - time_sim)
            box_vel_norm = np.linalg.norm(np.array(box_vel))
            box_position_prev = response.objects['Box'].objectSensors[0].position.arr

            robot.q = np.array(response.objects['robot'].angleSensors)
            iiwa.set_to_joint_position(robot.q)

            time_now = time.time()
            time_prev = time_now
            time_sim_prev = time_sim

            if (is_hit and box_vel_norm < 0.001 and time_now - time_init > contactTime+3):
                print("END")

                message = MessageFactory.create_controlmessage()
                robot_msg = message.objects["robot"]
                robot_msg.angleVelocities.extend([0, 0, 0, 0, 0, 0, 0])
                client.send(message)
                response = client.recv()

                break


# sudo python3 ../run-in-docker.py python3 click_application.py --model models/Projects/i_am_project/Scenes/IiwaPybullet.yml --timeStep 0.005 --agxOnly --rcs --portRange 5656 5658  --disableClickSync
