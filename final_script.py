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
from controller import get_joint_velocities_qp_dir_inertia_specific_NS, get_joint_velocities_qp
from get_robot import sim_robot_env
from iiwa_environment import object
from iiwa_environment import physics as phys
import functions as f


def read_model_data(model_data_path):
    # Open the HDF5 file in read mode
    hf = h5py.File(model_data_path, 'r')

    # Read the parameters
    n_components = hf['n_components'][()]
    means = hf['means'][()]
    covariances = hf['covariances'][()]
    weights = hf['weights'][()]

    # Close the HDF5 file
    hf.close()

    return n_components, means, covariances, weights


def integral_intersection_area(x, Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2):
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
    for i in range(n_components):
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
    for i in range(n_components):
        result1 += weights1[i] * (2 * np.pi * np.sqrt(dets_cov1[i])) * multivariate_normal.pdf([x[0], x[1]], mean=new_means1[i], cov=new_covariances1[i])
        result2 += weights2[i] * (2 * np.pi * np.sqrt(dets_cov2[i])) * multivariate_normal.pdf([x[0], x[1]], mean=new_means2[i], cov=new_covariances2[i])
    return result1


def fun(x, Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2):
    f = 0
    R2 = np.array([[np.cos(x[2]), -np.sin(x[2])],
                [np.sin(x[2]), np.cos(x[2])]])
    for i in range(n_components):
        R2 = np.squeeze(R2)
        mean = means2[i] + (x[:2]-box2)
        new_mean = R2 @ (mean-x[:2]) + x[:2]
        new_covariance = R2 @ covariances2[i] @ R2.T

        det_cov = np.linalg.det(new_covariance)

        pdf = weights2[i]*(2 * np.pi * np.sqrt(det_cov))* multivariate_normal.pdf(Xf, mean=new_mean, cov=new_covariance)

        f = f + pdf

    return -f*integral_intersection_area(x, Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2) #- 0.00*(integral_intersection_area(x) - intersection_threshold)

    # return -f- 0.2*(integral_intersection_area(x, Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2)-0.5)

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
                       Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2):
    
    fig, ax = plt.subplots()

    if colormap:
        x0_range = np.linspace(x_limits[0], x_limits[2], 100)
        x1_range = np.linspace(y_limits[0], y_limits[2], 100)

        pdf_values = np.zeros((len(x0_range), len(x1_range)))
        for i, x0_val in enumerate(x0_range):
            for j, x1_val in enumerate(x1_range):
                X1 = [x0_val, x1_val]
                pdf_values[j, i] = fun(X_opt, Xi, X1, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2)

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
    for i in range(n_components):
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

def output_results(X_opt,Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2):
    print("theta_1 = ", np.rad2deg(X_opt[3]))
    print("Xm = ", X_opt[:2])
    print("theta_2 = ", np.rad2deg(X_opt[2]))
    print("intersection = ", integral_intersection_area(X_opt,Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2))
    print("objective function = ", fun(X_opt, Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2))


def find_sol(environment,x_limits,y_limits,direction, intersection_threshold, margin,\
             Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2):
    if environment:
        cons = constraints(Xi, Xf, x_limits, y_limits, direction, margin)
    else:
        cons = ({'type': 'ineq', 'fun': lambda x:  integral_intersection_area(x) - intersection_threshold})

    guess_ = guess(Xi,Xf,direction)

    # HOW TO PASS THE KNOWN PARAMETERS (THAT WON'T NE OPTIMIZED FOR) TO fun??
    result = minimize(fun, guess_,args=(Xi, Xf, box1, box2, means1, covariances1, weights1, means2, covariances2, weights2),\
                      method='COBYLA', constraints=cons, tol=1e-8, options={'disp': True}) #, bounds = bnds)#, options={'tol': 1e-200}) # bounds= bnds)
    return result.x





model_data_path = 'Data/model.h5'
n_components, means, covariances, weights = read_model_data(model_data_path)

# Environment
box1 = np.array([0.5,0.3])
box2 = np.array([0.5,0.3])

Xi = [0.0,0.0]
Xf = [0.7,0.3]
x_limits = [-0.25, 0.25, 0.9]
y_limits = [-0.2, 0.2, 0.4]
table_direction = ['up','right']
environment = True
colormap = True
intersection_threshold = 0.6
margin = 0.05

X_opt = find_sol(environment,x_limits,y_limits,table_direction, intersection_threshold, margin,\
             Xi, Xf, box1, box2, means, covariances, weights, means, covariances, weights)

output_results(X_opt, Xi, Xf, box1, box2, means, covariances, weights, means, covariances, weights)

plot_optimization(X_opt, environment, x_limits, y_limits, table_direction, colormap,\
                       Xi, Xf, box1, box2, means, covariances, weights, means, covariances, weights)