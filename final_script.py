import numpy as np

import h5py
import matplotlib.pyplot as plt
import ellipse

from sklearn.model_selection import GridSearchCV
from scipy.stats import multivariate_normal
from scipy.stats import chi2
from scipy.optimize import minimize, basinhopping, Bounds
from scipy.integrate import nquad

from gmr import GMM

import os
import time

from ds import linear_hitting_ds_pre_impact, linear_ds
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp_dir_inertia_specific_NS2,\
     get_joint_velocities_qp
from get_robot_new import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
import functions as f


def read_model_data(model_data_path):
    # Open the HDF5 file in read mode
    hf_1 = h5py.File(model_data_path[0], 'a')
    hf_2 = h5py.File(model_data_path[1], 'a')

    # Read the parameters
    n_components_full = hf_1['n_components'][()]
    means_full = hf_1['means'][()]
    covariances_full = hf_1['covariances'][()]
    weights_full = hf_1['weights'][()]

    n_components_2d = hf_2['n_components'][()]
    means_2d = hf_2['means'][()]
    covariances_2d = hf_2['covariances'][()]
    weights_2d = hf_2['weights'][()]


    # Close the HDF5 file
    hf_1.close()
    hf_2.close()

    return n_components_full, means_full, covariances_full, weights_full,\
            n_components_2d, means_2d, covariances_2d, weights_2d


def integral_intersection_area(x, Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2):
    R1 = np.array([[np.cos(x[3]), -np.sin(x[3])],
                [np.sin(x[3]), np.cos(x[3])]])

    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])

    new_means1=[]
    new_covariances1=[]
    dets_cov1 = []
    new_means2=[]
    new_covariances2=[]
    dets_cov2 = []
    for i in range(n_components1):
        R1 = np.squeeze(R1)
        mean1 = means1[i] + (Xi-box1)
        new_mean1 = R1 @ (mean1-Xi) + Xi
        new_covariance1 = R1 @ covariances1[i] @ R1.T
        det_cov1 = np.linalg.det(new_covariance1)
        new_means1.append(new_mean1)
        new_covariances1.append(new_covariance1)
        dets_cov1.append(det_cov1)

        R2 = np.squeeze(R2)
        mean2 = means2[i] + (x[:2]-box2)
        new_mean2 = R2 @ (mean2-x[:2]) + x[:2]
        new_covariance2 = R2 @ covariances2[i] @ R2.T
        det_cov2 = np.linalg.det(new_covariance2)
        new_means2.append(new_mean2)
        new_covariances2.append(new_covariance2)
        dets_cov2.append(det_cov2)

    result1 = 0
    result2 = 0
    for i in range(n_components1):
        result1 += weights1[i] * (2 * np.pi * np.sqrt(dets_cov1[i])) * multivariate_normal.pdf([x[0], x[1]], mean=new_means1[i], cov=new_covariances1[i])
        result2 += weights2[i] * (2 * np.pi * np.sqrt(dets_cov2[i])) * multivariate_normal.pdf([x[0], x[1]], mean=new_means2[i], cov=new_covariances2[i])
    return result1


def fun(x, Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2):
    f = 0
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    for i in range(n_components1):
        R2 = np.squeeze(R2)
        mean = means2[i] + (x[:2]-box2)
        new_mean = R2 @ (mean-x[:2]) + x[:2]
        new_covariance = R2 @ covariances2[i] @ R2.T

        det_cov = np.linalg.det(new_covariance)

        pdf = weights2[i]*(2 * np.pi * np.sqrt(det_cov))* multivariate_normal.pdf(Xf, mean=new_mean, cov=new_covariance)

        f = f + pdf

    return -f*integral_intersection_area(x, Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2) #- 0.00*(integral_intersection_area(x) - intersection_threshold)

    # return -f- 0.2*(integral_intersection_area(x, Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                            #    n_components2, means2, covariances2, weights2)-0.5)

def constraints(Xi, Xf, x_limits, y_limits, direction, margin):
    cons = []

    # Box always on Table
    def cons_1(x, alpha, Xf, x_limits, direction, margin):
        if direction[1] == 'right':
            return ((1 - alpha) * x[0] + alpha * Xf[0]) - (x_limits[0] + margin)
        else:
            if direction[0] == 'up':
                return ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[0]+margin) if (((1 - alpha) * x[1] + alpha * Xf[1])>y_limits[1]) \
                    else ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[1]+margin)
            else:
                return ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[0]+margin) if (((1 - alpha) * x[1] + alpha * Xf[1])<y_limits[1]) \
                    else ((1 - alpha) * x[0] + alpha * Xf[0])-(x_limits[1]+margin)

    def cons_2(x, alpha, Xf, x_limits, y_limits, direction, margin):
        if direction[1] == 'right':
            if direction[0] == 'up':
                return -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[2]-margin)  if (((1 - alpha) * x[1] + alpha * Xf[1])>y_limits[1]) \
                    else -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[1]-margin)
            else:
                return -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[2]-margin)  if (((1 - alpha) * x[1] + alpha * Xf[1])<y_limits[1]) \
                    else -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[1]-margin)
        else:
            -((1 - alpha) * x[0] + alpha * Xf[0])+(x_limits[2]-margin)

    def cons_3(x, alpha, Xf, y_limits, direction, margin):
        if direction[0] == 'up':
            return -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[2]-margin)
        else:
            if direction[1] == 'right':
                return -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[2]-margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])<x_limits[1]) \
                    else -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[1]-margin)
            else:
                return -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[2]-margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])>x_limits[1]) \
                    else -((1 - alpha) * x[1] + alpha * Xf[1])+(y_limits[1]-margin)

    def cons_4(x, alpha, Xf, x_limits, y_limits, direction, margin):
        if direction[0] == 'up':
            if direction[1] == 'right':
                return ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[0]+margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])<x_limits[1]) \
                    else ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[1]+margin)
            else:
                return ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[0]+margin)  if(((1 - alpha) * x[0] + alpha * Xf[0])>x_limits[1]) \
                    else ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[1]+margin)
        else:
            return ((1 - alpha) * x[1] + alpha * Xf[1])-(y_limits[0]+margin)
        

    for a in np.linspace(0.0,1.0,num=30):
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_1(x, a, Xf, x_limits, direction, margin)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_2(x, a, Xf, x_limits, y_limits, direction, margin)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_3(x, a, Xf, y_limits, direction, margin)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_4(x, a, Xf, x_limits, y_limits, direction, margin)})

        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_1(x, a, Xi, x_limits, direction, margin)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_2(x, a, Xi, x_limits, y_limits, direction, margin)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_3(x, a, Xi, y_limits, direction, margin)})
        cons.append({'type': 'ineq', 'fun': lambda x, a=a: cons_4(x, a, Xi, x_limits, y_limits, direction, margin)})

    return cons

def guess(Xi,Xf, table_direction):
    guess = [0,0,0,0]

    if table_direction[0] == 'up' and table_direction[1] == 'right':
        guess[0] = (Xi[0]+Xf[0])/2
        guess[1] = (Xi[1]+Xf[1])/2
        guess[2] = np.arctan2(Xf[1]-guess[1],Xf[0]-guess[0])-np.pi/2
        guess[3] = np.arctan2(guess[1]-Xi[1],guess[0]-Xi[0]) -np.pi/2
    elif table_direction[0] == 'up' and table_direction[1] == 'left':
        guess[0] = (Xi[0]+Xf[0])/2
        guess[1] = (Xi[1]+Xf[1])/2
        guess[2] = np.arctan2(guess[0]-Xf[0],Xf[1]-guess[1])#+np.pi/2
        guess[3] = np.arctan2(Xi[0]-guess[0],guess[1]-Xi[1]) #+np.pi/2
    elif table_direction[0] == 'down' and table_direction[1] == 'left':
        guess[0] = (Xi[0]+Xf[0])/2
        guess[1] = (Xi[1]+Xf[1])/2
        guess[2] = np.arctan2(guess[1]-Xf[1],guess[0]-Xf[0])+np.pi/2
        guess[3] = np.arctan2(Xi[1]-guess[1],Xi[0]-guess[0]) +np.pi/2
    else:
        guess[0] = (Xi[0]+Xf[0])/2
        guess[1] = (Xi[1]+Xf[1])/2
        guess[2] = np.arctan2(Xf[0]-guess[0],guess[1]-Xf[1])+np.pi
        guess[3] = np.arctan2(guess[0]-Xi[0],Xi[1]-guess[1]) +np.pi
    return guess


def plot_optimization(X_opt, environment, x_limits, y_limits, table_direction, colormap,\
                       Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2):
    
    fig, ax = plt.subplots()

    if colormap:
        x0_range = np.linspace(x_limits[0], x_limits[2], 100)
        x1_range = np.linspace(y_limits[0], y_limits[2], 100)

        pdf_values = np.zeros((len(x0_range), len(x1_range)))
        for i, x0_val in enumerate(x0_range):
            for j, x1_val in enumerate(x1_range):
                X1 = [x0_val, x1_val]
                pdf_values[j, i] = fun(X_opt, Xi, X1, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2)

        plt.imshow(pdf_values, extent=[x_limits[0], x_limits[2], y_limits[0], y_limits[2]], origin='lower', cmap='viridis')

    # Plot initial position of box
    ax.scatter(Xi[0],Xi[1], s=100, marker='+',color='g', label ='Xi')
    ax.scatter(X_opt[0],X_opt[1], s=100, marker='+',color='b', label ='Xm')

    ax.plot([Xi[0],X_opt[0]],[Xi[1],X_opt[1]],color='k', linestyle='dashed')
    ax.plot([X_opt[0],Xf[0]],[X_opt[1],Xf[1]],color='k', linestyle='dashed')

    if environment:
        if table_direction[0] == 'up' and table_direction[1] == 'right':
            ax.plot(x_limits[0]*np.ones(2),[y_limits[0],y_limits[2]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[0],y_limits[1]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[1],y_limits[2]],color='k')

            ax.plot([x_limits[0],x_limits[2]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[1],x_limits[2]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[1]],y_limits[0]*np.ones(2),color='k', label ='Table')
        elif table_direction[0] == 'up' and table_direction[1] == 'left':
            ax.plot(x_limits[0]*np.ones(2),[y_limits[1],y_limits[2]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[0],y_limits[1]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[0],y_limits[2]],color='k')

            ax.plot([x_limits[0],x_limits[2]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[1]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[1],x_limits[2]],y_limits[0]*np.ones(2),color='k', label ='Table')
        elif table_direction[0] == 'down' and table_direction[1] == 'left':
            ax.plot(x_limits[0]*np.ones(2),[y_limits[0],y_limits[1]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[1],y_limits[2]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[0],y_limits[2]],color='k')

            ax.plot([x_limits[1],x_limits[2]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[1]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[2]],y_limits[0]*np.ones(2),color='k', label ='Table')
        else:
            ax.plot(x_limits[0]*np.ones(2),[y_limits[0],y_limits[2]],color='k')
            ax.plot(x_limits[1]*np.ones(2),[y_limits[1],y_limits[2]],color='k')
            ax.plot(x_limits[2]*np.ones(2),[y_limits[0],y_limits[1]],color='k')

            ax.plot([x_limits[0],x_limits[1]],y_limits[2]*np.ones(2),color='k')
            ax.plot([x_limits[1],x_limits[2]],y_limits[1]*np.ones(2),color='k')
            ax.plot([x_limits[0],x_limits[2]],y_limits[0]*np.ones(2),color='k', label ='Table')


    R1 = np.array([[np.cos(X_opt[3]), -np.sin(X_opt[3])],
                [np.sin(X_opt[3]), np.cos(X_opt[3])]])

    R2 = np.array([[np.cos(X_opt[2]), -np.sin(X_opt[2])],
                [np.sin(X_opt[2]), np.cos(X_opt[2])]])

    new_means1=[]
    new_covariances1=[]
    new_means2=[]
    new_covariances2=[]
    for i in range(n_components1):
        R1 = np.squeeze(R1)
        mean1 = means1[i] + (Xi-box1)
        new_mean1 = R1 @ (mean1-Xi) + Xi
        new_covariance1 = R1 @ covariances1[i] @ R1.T
        new_means1.append(new_mean1)
        new_covariances1.append(new_covariance1)

        R2 = np.squeeze(R2)
        mean2 = means2[i] + (X_opt[:2]-box2)
        new_mean2 = R2 @ (mean2-X_opt[:2]) + X_opt[:2]
        new_covariance2 = R2 @ covariances2[i] @ R2.T
        new_means2.append(new_mean2)
        new_covariances2.append(new_covariance2)


        ellipse.plot_ellipse(new_mean1,new_covariance1,ax)
        ellipse.plot_ellipse(new_mean2,new_covariance2,ax)

    ax.scatter(Xf[0],Xf[1], s=100, marker='+',color='k', label ='Xf')


    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    plt.title(f'Optimal setup to reach Xf = {Xf}')


    ax.legend()
    plt.show()

def output_results(X_opt,Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2):
    print("theta_1 = ", np.rad2deg(X_opt[3]))
    print("Xm = ", X_opt[:2])
    print("theta_2 = ", np.rad2deg(X_opt[2]))
    print("intersection = ", integral_intersection_area(X_opt,Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2))
    print("objective function = ", fun(X_opt, Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2))


def find_sol(environment,x_limits,y_limits,direction, intersection_threshold, margin,\
             Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2):
    if environment:
        cons = constraints(Xi, Xf, x_limits, y_limits, direction, margin)
    else:
        cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - intersection_threshold})

    guess_ = guess(Xi,Xf,direction)

    # HOW TO PASS THE KNOWN PARAMETERS (THAT WON'T NE OPTIMIZED FOR) TO fun??
    result = minimize(fun, guess_,args=(Xi, Xf, box1, box2, n_components1, means1, covariances1, weights1,\
                               n_components2, means2, covariances2, weights2),\
                      method='COBYLA', constraints=cons, tol=1e-8, options={'disp': True}) #, bounds = bnds)#, options={'tol': 1e-200}) # bounds= bnds)
    return result.x


def regress(n_components, means_, covariances_, weights_, known_parameters, to_predict='hitting_parameters'):
    # REGRESSION
    gmm = GMM(n_components=n_components, priors=weights_, means=means_,\
                covariances=np.array([c for c in covariances_]))
    
    if to_predict == 'final_position':
        known_parameters_index = [0,1,2,3]                       #Parameters
        x2_predicted_mean = gmm.predict(known_parameters_index, known_parameters)

        x_f = x2_predicted_mean[0][0]
        y_f = x2_predicted_mean[0][1]
        z_f = x2_predicted_mean[0][2]

        return x_f, y_f, z_f
    else:
        known_parameters_index = [4,5,6]                  #Positions
        x2_predicted_mean = gmm.predict(known_parameters_index, known_parameters)

        h_dir = np.array([0, 1, 0])
        p_des = x2_predicted_mean[0][0]
        x_impact = x2_predicted_mean[0][1]
        y_impact = x2_predicted_mean[0][2]
        z_impact = x2_predicted_mean[0][3]
        X_ref = [x_impact,y_impact,z_impact]
        
        return h_dir, p_des, X_ref

def simulate(X_i_robot, theta_i_robot, X_m_robot, theta_m_robot, h_dir1, p_des1, X_ref1, box_i, h_dir2, p_des2, X_ref2):
    trailDuration = 0 # Make it 0 if you don't want the trail to end
    contactTime = 0.5 # This is the time that the robot will be in contact with the box

    startPos1 = [X_i_robot[0], X_i_robot[1], 0]
    startPos2 = [X_m_robot[0], X_m_robot[1], 0]


    ################## GET THE ROBOT + ENVIRONMENT #########################
    box = object.Box([0.2, 0.2, 0.2], 0.5)  # the box is a cube of size 20 cm, and it is 0.5 kg in mass
    iiwa = sim_robot_env(1, box, startPos1, theta_i_robot, startPos2, theta_m_robot, box_i)

    ###################### INIT CONDITIONS #################################
    X_init = [0.3, -0.2, 0.5]
    q_init = iiwa.get_IK_joint_position(X_init) # Currently I am not changing any weights here
    Lambda_init = iiwa.get_inertia_matrix_specific(q_init)

    box_position_orientation = iiwa.get_box_position_orientation()
    box_position_init = box_position_orientation[0]
    box_orientation_init = box_position_orientation[1]

    A = np.array([[-2, 0, 0], [0, -2, 0], [0, 0, -2]])

    # initialize the robot and the box
    iiwa.set_to_joint_position(q_init)
    iiwa.set_to_joint_position2(q_init)
    iiwa.reset_box(box_position_init, box_orientation_init)
    lambda_dir = h_dir1.T @ Lambda_init @ h_dir1
    lambda_dir2 = h_dir2.T @ Lambda_init @ h_dir2

    is_hit1 = False

    # take some time
    time.sleep(1)    # !!!!

    # initialise the time
    time_init = time.time()


    is_hit2 = False
    # Start the motion
    while 1:
        print(np.array(iiwa.get_box_position_orientation()[0]))

        X_qp = np.array(iiwa.get_ee_position())
        
        if not is_hit1:
            dX = linear_hitting_ds_pre_impact(A, X_qp, X_ref1, h_dir1, p_des1, lambda_dir, box.mass)
        else:
            dX = linear_ds(A, X_qp, X_ref1)

        hit_dir = dX / np.linalg.norm(dX)

        lambda_current = iiwa.get_inertia_matrix()
        lambda_dir = hit_dir.T @ lambda_current @ hit_dir
        
        jac = np.array(iiwa.get_trans_jacobian())
        q_dot = get_joint_velocities_qp_dir_inertia_specific_NS(dX, jac, iiwa, hit_dir, 0.15, lambda_dir)
        
        iiwa.move_with_joint_velocities(q_dot)

        ## Need something more here later, this is contact detection and getting the contact point
        if(iiwa.get_collision_points().size != 0):
            is_hit1 = True
            iiwa.get_collision_position()

            
        iiwa.step()
        time_now = time.time()

        # if((is_hit and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime) or (time_now - time_init > 10)):
        #     box_pos = np.array(iiwa.get_box_position_orientation()[0])
        #     break

        if (is_hit1 and (time_now - time_init) > 1.5 and iiwa.get_box_speed() < 0.5):
        #if (iiwa.get_box_position_orientation()[0][1]>0.5):
            #x_b = np.array(iiwa.get_box_position_orientation()[0]) + np.array([0,0.2,0])
            while 1:
                print(np.array(iiwa.get_box_position_orientation()[0]))
                X_qp2 = np.array(iiwa.get_ee_position2())  #########
                
                x_b = np.array(iiwa.get_box_position_orientation()[0])
                
                if not is_hit2:
                    dX = linear_hitting_ds_pre_impact(A, X_qp2, X_ref2, h_dir2, 1.0, lambda_dir2, box.mass)
                else:
                    dX = linear_ds(A, X_qp2, X_ref2)
    
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
        
                iiwa.step()
                time_now = time.time()

            #     if(is_hit1 and is_hit2 and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime):
            #         break
            # if(is_hit1 and is_hit2 and iiwa.get_box_speed() < 0.001 and time_now - time_init > contactTime):
            #     break

def rot_pts(X, pivot, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]])
    X_rot = R @ (X - pivot) + pivot
    return X_rot



def simulation_parameters(X_opt, Xi, Xf, box1, box2):
    theta_i_robot = X_opt[3]
    theta_m_robot = X_opt[2]

    # theta_i_star = theta_i_robot - box_robot_angle
    # X_i_robot = Xi-box_robot_dist*np.array([-np.sin(theta_i_star),np.cos(theta_i_star)])
    # print("X_i_robot", X_i_robot)


    box_robot_dist = np.sqrt(box1[0]**2 + box1[1]**2)
    box_robot_angle = np.arctan2(box1[0],box1[1])

    # Robot 1 position calculated from Xi
    Xi_robot0 = Xi - box_robot_dist*np.array([np.sin(box_robot_angle),np.cos(box_robot_angle)])
    # Robot 1 position rotated
    X_i_robot = rot_pts(Xi_robot0, Xi, theta_i_robot)

    X_m =  X_opt[:2]

    # Robot 2 position calculated from Xm
    X_m_robot0 = X_m - box_robot_dist*np.array([np.sin(box_robot_angle),np.cos(box_robot_angle)])
    # Robot 2 position rotated
    X_m_robot = rot_pts(X_m_robot0, X_m, theta_m_robot)


    # COLLECT DATA WITHOUT TABLE !!!!!

    # Bring Xm and Xf to reference of reachable space (to regress)
    X_m_rot = rot_pts(X_m, Xi, -theta_i_robot) - (Xi - box1)
    X_f_rot = rot_pts(Xf, X_m, -theta_m_robot) - (X_m - box2)

    ### Determine hitting parameters to reach Xm
    x_reg = X_m_rot[0]
    y_reg = X_m_rot[1]
    z_reg = 0.44922494
    known_parameters = [[x_reg, y_reg, z_reg]]  
    h_dir1, p_des1, X_ref1 = regress(n_components_full, means_full, covariances_full, weights_full, known_parameters)
    # Rotate back hitting point to real scenario
    X_ref_rot1 = rot_pts(np.array(X_ref1[:2]), Xi, theta_i_robot)
    X_ref1[:2] = X_ref_rot1
    X_ref1[2] = X_ref1[2]-0.4   # to be on the floor (HAVE TO CHANGE)
    # Rotate back hitting direction to real scenario
    h_dir_rot1 = []
    h_dir_rot1_ = rot_pts(np.array(h_dir1[:2]), 0, theta_i_robot)
    h_dir_rot1 = [0.0,0.0,0.0]
    h_dir_rot1[:2] = h_dir_rot1_
    h_dir_rot1 = np.array(h_dir_rot1)

    ### Determine hitting parameters to reach Xf
    x_reg = X_f_rot[0]
    y_reg = X_f_rot[1]
    z_reg = 0.44922494
    known_parameters = [[x_reg, y_reg, z_reg]]  
    h_dir2, p_des2, X_ref2 = regress(n_components_full, means_full, covariances_full, weights_full, known_parameters)
    # Rotate back hitting point to real scenario
    X_ref_rot2 = rot_pts(np.array(X_ref2[:2]), X_m, theta_m_robot)
    X_ref2[:2] = X_ref_rot2
    X_ref2[2] = X_ref2[2]-0.4    # to be on the floor (HAVE TO CHANGE)
    # Rotate back hitting direction to real scenario
    h_dir_rot2 = []
    h_dir_rot2_ = rot_pts(np.array(h_dir2[:2]), 0, theta_m_robot)
    h_dir_rot2 = [0.0,0.0,0.0]
    h_dir_rot2[:2] = h_dir_rot2_
    h_dir_rot2 = np.array(h_dir_rot2)

    return X_i_robot, theta_i_robot, X_m_robot, theta_m_robot, h_dir_rot1, p_des1, X_ref1, h_dir_rot2, p_des2, X_ref2



model_data_paths = ['Data/model_no_table_full.h5', 'Data/model_no_table_2d.h5']
n_components_full, means_full, covariances_full, weights_full,\
            n_components_2d, means_2d, covariances_2d, weights_2d = read_model_data(model_data_paths)


# # COLLECT DATA WITHOUT TABLE !!!!!

# x_reg = 0.6 #0.49751953
# y_reg = 1.0 #0.49648312
# z_reg = 0.44922494
# known_parameters = [[x_reg, y_reg, z_reg]]  
# h_dir, p_des, X_ref = regress(n_components_full, means_full, covariances_full, weights_full, known_parameters)
# X_ref[2] = X_ref[2]-0.4   # HAVE TO CHANGE


# p_des_reg = 0.9
# x_impact_reg = 0.5
# y_impact_reg = 0.3
# z_impact_reg = 0.5
# known_parameters = [[p_des_reg, x_impact_reg, y_impact_reg,z_impact_reg]]  #Parameters
# x_f, y_f, z_f = regress(n_components_full, means_full, covariances_full, weights_full, known_parameters, to_predict='final_position')


# Environment
box1 = np.array([0.5,0.3])
box2 = np.array([0.5,0.3])


Xi = np.array([0.5,0.3])
Xf = [1.2,0.6]
x_limits = [0.25, 0.6, 1.4]
y_limits = [0.1, 0.5, 0.7]
table_direction = ['up','right']
environment = True
colormap = False
intersection_threshold = 0.6
margin = 0.05

X_opt = find_sol(environment,x_limits,y_limits,table_direction, intersection_threshold, margin,\
             Xi, Xf, box1, box2, n_components_2d, means_2d, covariances_2d, weights_2d,\
                 n_components_2d, means_2d, covariances_2d, weights_2d)

output_results(X_opt, Xi, Xf, box1, box2, n_components_2d, means_2d, covariances_2d, weights_2d,\
               n_components_2d, means_2d, covariances_2d, weights_2d)

plot_optimization(X_opt, environment, x_limits, y_limits, table_direction, colormap,\
                       Xi, Xf, box1, box2, n_components_2d, means_2d, covariances_2d, weights_2d,\
                         n_components_2d, means_2d, covariances_2d, weights_2d)

X_i_robot, theta_i_robot, X_m_robot,theta_m_robot,\
      h_dir_rot1, p_des1, X_ref1, h_dir_rot2, p_des2, X_ref2 = simulation_parameters(X_opt, Xi, Xf, box1, box2)


simulate(X_i_robot, theta_i_robot, X_m_robot, theta_m_robot, h_dir_rot1, p_des1, X_ref1, Xi, h_dir_rot2, p_des2, X_ref2)
