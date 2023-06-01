import numpy as np
import scipy as sp
import scipy.linalg


def get_stein_divergence(A_1, A_d):
    return np.log(np.abs(np.linalg.det(0.5 * (A_d + A_1)))) - 0.5 * np.log(np.abs(np.linalg.det(A_d @ A_1)))

def get_stein_divergence_gradient(A_1, A_d):
    return 0.5 * (np.linalg.inv((A_d + A_1) / 2) - np.linalg.inv(A_d))

def get_gradient_projection(A, dA):
    return 0.5 * A @ (dA + dA.T) @ A

def get_exp_retraction(A, dA_tangent):
    A_sqrt = sp.linalg.sqrtm(A)
    A_sqrt_inv = np.linalg.inv(A_sqrt)
    return A_sqrt @ np.array(sp.linalg.expm(A_sqrt_inv @ dA_tangent @ A_sqrt_inv)) @ A_sqrt

def contact_detection():
    contact = False
    ## Here code needs to be written for contact detection
    return contact


#######################################################################################

def des_hitting_point(box_object, init_position):
    # Let's start with the center of the box
    x_length = box_object.l
    y_length = box_object.b
    z_length = box_object.h
    X_hit = init_position + np.array([0.0, 0.0, 0.0])
    return X_hit

def des_hitting_point_grid(box_object, init_position, face, grid_size):
    # Let's start with the center of the box
    x_length = box_object.l
    y_length = box_object.b
    z_length = box_object.h
    X_hit = np.tile(init_position, (grid_size*grid_size, 1))

    N = grid_size*grid_size

    x_l = np.linspace(-x_length/2+0.1*x_length, x_length/2-0.1*x_length, grid_size)
    y_l = np.linspace(-y_length/2+0.1*y_length, y_length/2-0.1*y_length, grid_size)
    z_l = np.linspace(-z_length/2+0.1*z_length, z_length/2-0.1*z_length, grid_size)

    grid_02 = np.array(np.meshgrid(x_l, z_l))
    hit_grid_02 = np.resize(grid_02, (2, grid_size*grid_size))
    grid_13 = np.array(np.meshgrid(y_l, z_l))
    hit_grid_13 = np.resize(grid_13, (2, grid_size*grid_size))
    grid_45 = np.array(np.meshgrid(x_l, y_l))
    hit_grid_45 = np.resize(grid_45, (2, grid_size*grid_size))

    # For the 6 different faces of the box numbered from 0 to 5
    # Create the grid
    if face == 0 or face == 2:
        for i in range(N):
            X_hit[i, :] = X_hit[i, :] + np.array([hit_grid_02[0, i], 0.0, hit_grid_02[1, i]])
    if face == 1 or face == 3:
        for i in range(grid_size):
            X_hit[i, :] = X_hit[i, :] + np.array([0.0, hit_grid_13[0, i], hit_grid_13[1, i]])
    if face == 4 or face == 5:
        for i in range(grid_size):
            X_hit[i,:] = X_hit[i,:] + np.array([hit_grid_45[0, i], hit_grid_45[1, i], 0.0])
    return X_hit