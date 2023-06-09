import numpy as np
import os
import time
import h5py

from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
from get_robot import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
import functions as f


hf = h5py.File('Data/data_no_table_2.h5', 'w')
group = hf.create_group("my_data")


# params_ds = group.create_dataset(f"params_{X_ref}", shape=(0, 4), maxshape=(None, 4), dtype=np.float64)
# box_pos_ds = group.create_dataset(f"box_pos_{X_ref}", shape=(0, 3), maxshape=(None, 3), dtype=np.float64)


######################### PARAMETERS ###############################
trailDuration = 0 # Make it 0 if you don't want the trail to end
contactTime = 0.5 # This is the time that the robot will be in contact with the box

################## GET THE ROBOT + ENVIRONMENT #########################
box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass
iiwa = sim_robot_env(1, box)

###################### INIT CONDITIONS #################################
X_init = [0.3, -0.2, 0.2]
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

# p_des grid
p_des_grid = np.linspace(0.7,1,4)

# Velocity grid

########################################################################

params_dataset = group.create_dataset("params", (len(X_ref_grid)*len(p_des_grid), 4), dtype='f')
box_pos_dataset = group.create_dataset("box_pos", (len(X_ref_grid)*len(p_des_grid), 3), dtype='f')

i = 0


for p_des in p_des_grid:
    print("p_des: ", p_des)
    for X_ref in X_ref_grid:
        
        #print("X_ref: ", X_ref)

        # initialize the robot and the box
        iiwa.set_to_joint_position(q_init)
        iiwa.reset_box(box_position_init, box_orientation_init)
        lambda_dir = h_dir.T @ Lambda_init @ h_dir
        is_hit = False

        # take some time
        time.sleep(1)    # !!!!

        # initialise the time
        time_init = time.time()

        # Start the motion
        while 1:
            X_qp = np.array(iiwa.get_ee_position())
            
            if not is_hit:
                dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref, h_dir, p_des, lambda_dir, box.mass)
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

            
            params = np.array([p_des,X_ref[0],X_ref[1],X_ref[2]])
            
            iiwa.step()
            time_now = time.time()

            if((is_hit and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime)):# or (time_now - time_init > 10)):
                box_pos = np.array(iiwa.get_box_position_orientation()[0])
                # # Append the data to the datasets
                params_dataset[i:i+1] = params.reshape(1, 4)
                box_pos_dataset[i:i+1] = box_pos.reshape(1, 3)
                i = i + 1
                
                # params_ds.resize((params_ds.shape[0] + 1, 4))
                # params_ds[-1] = params
                
                # box_pos_ds.resize((box_pos_ds.shape[0] + 1, 3))
                # box_pos_ds[-1] = box_pos
                break
hf.close()
