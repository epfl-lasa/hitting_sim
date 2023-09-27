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


def modulation_matrix(E, V):
    return E @ V @ E.T


def sphere_modulation_centre(position, radius):
    '''
    Places a sphere at the base of the robot

    The robot end effector coming too close to the robot, increases the inertia
    extremely in the x, y directions and z is high
    '''
    centre = np.array([0, 0, 0])
    dist = np.square(np.linalg.norm(position - centre)) - np.square(radius) + 1

    # print(dist)

    e1 = position - centre
    e1 = np.reshape(e1, (3, 1))
    e1 = -e1 / np.linalg.norm(e1)
    e2 = np.zeros((3, 1))
    e2[1] = -e1[2]
    e2[2] = e1[1]
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1.reshape((3,)), e2.reshape((3,)))
    e3 = e3 / np.linalg.norm(e3)
    e3 = np.reshape(e3, (3, 1))

    E = np.hstack((e1, e2, e3))
    v1 = 1 - 1/np.abs(dist)
    v2 = 1 + 1/np.abs(dist)
    V = np.diag(np.array([v1, v2, v2]))

    return E @ V @ E.T


def cylinder_modulation_centre(position, radius):
    '''
    Places a sphere at the base of the robot

    The robot end effector coming too close to the robot, increases the inertia
    extremely in the x, y directions and z is high
    '''
    centre = np.array([0, 0, position[2]])

    dist = np.square(np.linalg.norm(position - centre)) - np.square(radius) + 1

    # print(dist)

    e1 = position - centre
    e1 = np.reshape(e1, (3, 1))
    e1 = -e1 / np.linalg.norm(e1)
    e2 = np.zeros((3, 1))
    e2[1] = -e1[2]
    e2[2] = e1[1]
    e2 = e2 / np.linalg.norm(e2)
    e3 = np.cross(e1.reshape((3,)), e2.reshape((3,)))
    e3 = e3 / np.linalg.norm(e3)
    e3 = np.reshape(e3, (3, 1))

    E = np.hstack((e1, e2, e3))
    v1 = 1 - 1/np.abs(dist)
    v2 = 1 + 1/np.abs(dist)
    V = np.diag(np.array([v1, v2, v2]))

    return E @ V @ E.T


def inertia_modulation():
    return 0