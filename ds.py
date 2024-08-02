import numpy as np

def linear_ds(A, X, X_d):
    return A @ (X - X_d)


def second_order_ds(A1, A2, X, X_d, X_dot):
    return A1 @ (X - X_d) + A2 @ X_dot


def linear_hitting_ds(A1, X, X_obj, v_hit):
    '''
    '''
    obj_virtual = X_obj + np.dot((X - X_obj), v_hit) * v_hit / np.square(np.linalg.norm(v_hit))
    sigma = 0.1
    alpha = np.exp(-np.linalg.norm(X - obj_virtual)/np.square(sigma))
    # print(alpha)
    dX = alpha * v_hit + (1 - alpha) * A1 @ (X - obj_virtual)
    return dX

def linear_hitting_ds_momentum(A1, X, X_obj, v_hit, p_des, lambda_current):
    '''
    '''
    obj_virtual = X_obj + np.dot((X - X_obj), v_hit) * v_hit / np.square(np.linalg.norm(v_hit))
    sigma = 0.1
    alpha = np.exp(-np.linalg.norm(X - obj_virtual)/np.square(sigma))
    dX = alpha * v_hit + (1 - alpha) * A1 @ (X - obj_virtual)
    dX = (p_des/lambda_current) * dX / np.linalg.norm(dX)
    return dX

def linear_hitting_ds_pre_impact(A1, X, X_obj, v_hit, p_des, lambda_current, m_obj):
    '''
    '''
    obj_virtual = X_obj + np.dot((X - X_obj), v_hit) * v_hit / np.square(np.linalg.norm(v_hit))
    sigma = 0.1
    alpha = np.exp(-np.linalg.norm(X - obj_virtual)/np.square(sigma))
    dX = alpha * v_hit + (1 - alpha) * A1 @ (X - obj_virtual)
    dX = (p_des/lambda_current)*(lambda_current+m_obj) * dX / np.linalg.norm(dX)
    return dX

def second_order_hitting_ds(A1, A2, X, X_obj, X_dot, v_des):
    '''
    The object here needs a virtualised motion
    virtual object is the projection of the end effector on the hitting direction
    hitting direction will be the same as the direction of the velocity!
    '''
    obj_virtual = X_obj + np.dot((X - X_obj), v_des) * v_des / np.square(np.linalg.norm(v_des))
    # print(obj_virtual)
    acc = A1 @ (X - obj_virtual) + A2 @ (X_dot - v_des)
    return acc


def linear_ds_momentum(A, X, X_d, dir_inertia, mom_des):
    fx = A @ (X - X_d)
    fx = fx / np.linalg.norm(fx)
    return (mom_des / dir_inertia) * fx
