import numpy as np
import os
import time

from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
from controller import get_joint_velocities_qp_dir_inertia_specific_NS2, get_joint_velocities_qp2
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
Lambda_init2 = iiwa.get_inertia_matrix_specific2(q_init)


##################### DS PROPERTIES ####################################
A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])
h_dir = np.array([0, 1, 0]) # This is the direction of the hitting
h_dir2 = np.array([0, -1, 0])

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
    iiwa.set_to_joint_position2(q_init)
    iiwa.reset_box(box_position_init, box_orientation_init)
    lambda_dir = h_dir.T @ Lambda_init @ h_dir
    lambda_dir2 = h_dir2.T @ Lambda_init2 @ h_dir2
    is_hit1 = False
    is_hit2 = False

    # take some time
    time.sleep(1)

    # initialise the time
    time_init = time.time()

    # Start the motion
    while 1:
        X_qp = np.array(iiwa.get_ee_position())
        
        if not is_hit1:
            dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, h_dir, 0.7, lambda_dir, box.mass)
        else:
            dX = linear_ds(A, X_qp, X_ref)

        hit_dir = dX / np.linalg.norm(dX)

        lambda_current = iiwa.get_inertia_matrix()
        lambda_dir = hit_dir.T @ lambda_current @ hit_dir
        
        jac = np.array(iiwa.get_trans_jacobian())
        q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, iiwa, hit_dir, 0.15, lambda_dir)
        
        iiwa.move_with_joint_velocities(q_dot)

        
        if(iiwa.get_collision_points().size != 0):
            is_hit1 = True
            iiwa.get_collision_position()
            
        iiwa.step()
        time_now = time.time()
        
        if ((time_now - time_init) > 1.5 and iiwa.get_box_speed() < 0.5):
        #if (iiwa.get_box_position_orientation()[0][1]>0.5):
            #x_b = np.array(iiwa.get_box_position_orientation()[0]) + np.array([0,0.2,0])
            while 1:
                X_qp = np.array(iiwa.get_ee_position2())
                
                x_b = np.array(iiwa.get_box_position_orientation()[0])
                
                if not is_hit2:
                    dX = linear_hitting_ds_pre_impact(A, X_qp, x_b, h_dir2, 0.7, lambda_dir2, box.mass)
                else:
                    dX = linear_ds(A, X_qp, X_ref)
    
                hit_dir = dX / np.linalg.norm(dX)
    
                lambda_current = iiwa.get_inertia_matrix2()
                lambda_dir2 = hit_dir.T @ lambda_current @ hit_dir
                
                jac = np.array(iiwa.get_trans_jacobian2())
                q_dot = get_joint_velocities_qp_dir_inertia_specific_NS2(dX, jac, iiwa, hit_dir, 0.15, lambda_dir2)
                
                iiwa.move_with_joint_velocities2(q_dot)
                
                ## Need something more here later, this is contact detection and getting the contact point
                if(iiwa.get_collision_points2().size != 0):
                    is_hit2 = True
                    iiwa.get_collision_position2()
                    
                if(is_hit1 and is_hit2 and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime):
                    break
            if(is_hit1 and is_hit2 and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime):
                break
        
        