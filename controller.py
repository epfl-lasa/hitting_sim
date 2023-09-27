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

from qpsolvers import solve_qp
from qpsolvers import solve_ls
import numpy as np

def get_joint_velocities(fx, jacobian, manipulator):
    R = jacobian
    s = fx
    return solve_ls(R, s, G=None, h=None, lb=manipulator.q_dot_ll, ub=manipulator.q_dot_ul, solver="osqp", eps_abs=1e-4)


def get_joint_velocities_qp(fx, jacobian, manipulator):
    P = jacobian.T @ jacobian
    fx = fx.reshape(3, 1)
    q_ = -1 * (jacobian.T @ fx)
    return solve_qp(P, q_, lb=manipulator.q_dot_ll, ub=manipulator.q_dot_ul, solver="osqp", eps_abs=1e-4)


def get_joint_velocities_qp_inertia(fx, jacobian, g_cur, A_d, manipulator):
    k = 0.01
    P = jacobian.T @ jacobian
    dg_dq = manipulator.get_stein_divergence_joint_gradient(g_cur, A_d)
    fx = fx.reshape(3, 1)
    q_ = -1 * (jacobian.T @ fx) + k * dg_dq

    return solve_qp(P, q_, lb=manipulator.q_dot_ll, ub=manipulator.q_dot_ul, solver="osqp", eps_abs=1e-4)

def get_joint_velocities_qp_inertia_NS(fx, jacobian, g_cur, A_d, manipulator):
    k = 0.01
    P = jacobian.T @ jacobian
    dg_dq = manipulator.get_stein_divergence_joint_gradient(g_cur, A_d)
    fx = fx.reshape(3, 1)
    N = (np.identity(7) - np.linalg.pinv(jacobian) @ jacobian)
    q_ = -1 * (jacobian.T @ fx) # + k * dg_dq
    q_dot_2 = - 0.0 * (N @ dg_dq)
    q_dot_1 = solve_qp(P, q_, lb=manipulator.q_dot_ll, ub=manipulator.q_dot_ul, solver="osqp", eps_abs=1e-4)
    q_dot_return = q_dot_1 + q_dot_2.reshape(7, )

    return q_dot_return

def get_joint_velocities_qp_weighted_inertia(fx, jacobian, g_cur, A_d, manipulator):
    k = np.diag(np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]))
    P = jacobian.T @ jacobian
    dg_dq = manipulator.get_stein_divergence_joint_gradient(g_cur, A_d)
    fx = fx.reshape(3, 1)
    q_ = -1 * (jacobian.T @ fx) + k @ dg_dq

    return solve_qp(P, q_, lb=manipulator.q_dot_ll, ub=manipulator.q_dot_ul, solver="osqp", eps_abs=1e-4)


def get_joint_velocities_qp_dir_inertia(fx, jacobian, manipulator, direction):
    k = 0.001
    P = jacobian.T @ jacobian
    dl_dq_dir = manipulator.get_directional_inertia_gradient(direction)
    fx = fx.reshape(3, 1)
    q_ = -1 * (jacobian.T @ fx) - k * dl_dq_dir

    return solve_qp(P, q_, lb=manipulator.q_dot_ll, ub=manipulator.q_dot_ul, solver="osqp", eps_abs=1e-4)

def get_joint_velocities_qp_dir_inertia_specific_NS(fx, jacobian, manipulator, direction, alpha, l_dir):

    q_dot_1 = get_joint_velocities_qp(fx, jacobian, manipulator)
    N = (np.identity(7) - np.linalg.pinv(jacobian) @ jacobian)
    dl_dq_dir = manipulator.get_directional_inertia_gradient(direction)
    q_dot_2 =  -alpha * (N @ dl_dq_dir * (l_dir - 6))
    q_print = q_dot_2/alpha
    q_dot_return = q_dot_1 + q_dot_2.reshape(7, )
    return q_dot_return



def get_joint_velocities_qp_dir_inertia_NS(fx, jacobian, manipulator, direction, alpha):

    q_dot_1 = get_joint_velocities_qp(fx, jacobian, manipulator)
    N = (np.identity(7) - np.linalg.pinv(jacobian) @ jacobian)
    dl_dq_dir = manipulator.get_directional_inertia_gradient(direction)
    # print(dl_dq_dir.T)
    q_dot_2 = alpha * (N @ dl_dq_dir)
    q_print = q_dot_2/alpha
    # print(q_print.T)
    q_dot_return = q_dot_1 + q_dot_2.reshape(7, )
    return q_dot_return


def inertia_control_qp(g_cur, A_d, manipulator):
    P = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]))
    k = 20
    dg_dq = manipulator.get_stein_divergence_joint_gradient(g_cur, A_d)
    q_ = k * dg_dq
    return solve_qp(P, q_, lb=manipulator.q_dot_ll, ub=manipulator.q_dot_ul, solver="osqp", eps_abs=1e-4)


def qp_controller(fx, jacobian, g_cur, A_d, manipulator, q_cur, dt):
    k = 1
    A = jacobian.T @ jacobian
    P = 1 * np.block([[A, np.zeros((7, 1))], [np.zeros((1, 7)), 0.1*np.identity(1).reshape((1, 1))]])
    dg_dq = manipulator.get_stein_divergence_joint_gradient(g_cur, A_d)
    fx = fx.reshape(3, 1)
    Q = -1 * (jacobian.T @ fx) + k * dg_dq
    q_ = np.vstack((Q, np.zeros((1, 1))))
    # print("P matrix ", P)
    # print("q matrix ", q_)
    lower_bound = np.hstack((manipulator.q_dot_ll, 0.0 * np.zeros(1)))
    upper_bound = np.hstack((manipulator.q_dot_ul, 100000 * np.ones(1)))
    g1_ = -dt * np.identity(7)
    g2_ = 0.0 * np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).reshape(7, 1)

    G1_ = np.hstack((g1_, g2_))
    G2_ = np.hstack((-g1_, g2_))

    G_ = np.vstack((G1_, G2_))

    h1_ = q_cur - manipulator.q_ll
    h2_ = - q_cur + manipulator.q_ul

    h_ = np.hstack((h1_, h2_))

    # print(h_)

    # print(P.shape, q_.shape, G_.shape, h_.shape, lower_bound.shape, manipulator.q_dot_ll.shape)
    q_vel = solve_qp(P, q_, G=G_, h=h_, lb=lower_bound, ub=upper_bound, solver="osqp", eps_abs=1e-2)
    # print(q_vel)
    return q_vel