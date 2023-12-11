import numpy as np
import os
import time
import yaml
import zmq

from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
from get_robot_env import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
import functions as f

####################################################################
######################## ZMQ CONNECTION ############################
####################################################################
with open('config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

context = zmq.Context()

''' I need to send the robot state and the box state to the Neural Network '''
socket_send_state = f.init_publisher(context, '*', config["zmq"]["state_port"])

######################### PARAMETERS ###############################
trailDuration = 0 # Make it 0 if you don't want the trail to end
contactTime = 0.5 # This is the time that the robot will be in contact with the box

################## GET THE ROBOT + ENVIRONMENT #########################
box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass
robot = sim_robot_env(1, box, 1)

##################### DS PROPERTIES ####################################
A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])
h_dir = np.array([0, 1, 0]) # This is the direction of the hitting

###################### DESIRED DIRECTIONAL PROPERTIES ##################
box_position_orientation = robot.get_box_position_orientation()
box_position_init = box_position_orientation[0]
box_orientation_init = box_position_orientation[1]
X_ref = f.des_hitting_point(box, box_position_init) # This needs to come from the box position
X_ref_grid = f.des_hitting_point_grid(box, box_position_init, 0, 5)

########################################################################

# initialize the robot and the box
robot.set_to_joint_position(robot.rest_pose)
robot.step()
Lambda_init = robot.get_inertia_matrix()
robot.reset_box(box_position_init, box_orientation_init)
lambda_dir = h_dir.T @ Lambda_init @ h_dir
is_hit = False

# take some time
time.sleep(1)

# initialise the time
time_init = time.time()

# Start the motion
while 1:
    X_qp = np.array(robot.get_ee_position())

    '''Follow the Hitting DS and then the Linear DS'''
    if not is_hit:
        dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, h_dir, 0.55, lambda_dir, box.mass)
    else:
        dX = linear_ds(A, X_qp, X_ref)

    hit_dir = dX / np.linalg.norm(dX)

    lambda_current = robot.get_inertia_matrix()
    lambda_dir = hit_dir.T @ lambda_current @ hit_dir
    jac = np.array(robot.get_trans_jacobian())

    '''The different DS are controlled differently'''
    if not is_hit:
        q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, robot, hit_dir, 0.15, lambda_dir)
    else:
        q_dot = get_joint_velocities_qp(dX, jac, robot)
    
    
    robot.move_with_joint_velocities(q_dot)

    # ## Need something more here later, this is contact detection and getting the contact point
    if(robot.get_collision_points().size != 0):
        is_hit = True
        robot.get_collision_position()


    robot.step()
    time_now = time.time()

    # if(is_hit and robot.get_box_speed() < 0.001 and time_now - time_init > contactTime):
    #     break