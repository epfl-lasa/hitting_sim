#|
#|    Copyright (C) 2021-2023 Learning Algorithms and Systems Laboratory, EPFL, Switzerland
#|    Authors: Harshit Khurana (maintainer)
#|
#|    email:   harshit.khurana@epfl.ch
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
import os
import time

from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
from get_robot import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
import functions as f

######################### PARAMETERS ###############################
trailDuration = 0 # Make it 0 if you don't want the trail to end
contactTime = 0.5 # This is the time that the robot will be in contact with the box

################## GET THE ROBOT + ENVIRONMENT #########################
box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass
iiwa = sim_robot_env(1, box)

###################### INIT CONDITIONS #################################
X_init = [0.3, -0.2, 0.5]
q_init = iiwa.get_IK_joint_position(X_init) # Currently I am not changing any weights here
Lambda_init = iiwa.get_inertia_matrix_specific(q_init)

##################### DS PROPERTIES ####################################
A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])
h_dir = np.array([0, 1, 0]) # This is the direction of the hitting

###################### DESIRED DIRECTIONAL PROPERTIES ##################
box_position_orientation = iiwa.get_box_position_orientation()
box_position_init = box_position_orientation[0]
box_orientation_init = box_position_orientation[1]
# X_ref = f.des_hitting_point(box, box_position_init) # This needs to come from the box position
X_ref_grid = f.des_hitting_point_grid(box, box_position_init, 0, 5)

########################################################################


for X_ref in X_ref_grid:

    print("X_ref: ", X_ref)

    # initialize the robot and the box
    iiwa.set_to_joint_position(q_init)
    iiwa.reset_box(box_position_init, box_orientation_init)
    lambda_dir = h_dir.T @ Lambda_init @ h_dir
    is_hit = False

    # take some time
    time.sleep(1)

    # initialise the time
    time_init = time.time()

    # Start the motion
    while 1:
        X_qp = np.array(iiwa.get_ee_position())
        
        if not is_hit:
            dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, h_dir, 0.7, lambda_dir, box.mass)
        else:
            dX = linear_ds(A, X_qp, X_ref)

        hit_dir = dX / np.linalg.norm(dX)

        lambda_current = iiwa.get_inertia_matrix()
        lambda_dir = hit_dir.T @ lambda_current @ hit_dir
        
        jac = np.array(iiwa.get_trans_jacobian())
        q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, iiwa, hit_dir, 0.15, lambda_dir)
        
        iiwa.move_with_joint_velocities(q_dot)

        ## Need something more here later, this is contact detection and getting the contact point
        if(iiwa.get_collision_points().size != 0):
            is_hit = True
            iiwa.get_collision_position()


        iiwa.step()
        time_now = time.time()

        if(is_hit and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime):
            break