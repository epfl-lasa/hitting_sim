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

def get_joint_velocities_qp_dir_inertia_specific_NS(fx, jacobian, manipulator, direction, alpha, l_dir, l_des):

    q_dot_1 = get_joint_velocities_qp(fx, jacobian, manipulator)
    N = (np.identity(7) - np.linalg.pinv(jacobian) @ jacobian)
    dl_dq_dir = manipulator.get_effective_inertia_gradient(direction)
    # dl_dq_dir = manipulator.get_directional_inertia_gradient(direction)
    q_dot_2 =  -alpha * (N @ dl_dq_dir * (l_dir - l_des))
    q_dot_return = q_dot_1 + q_dot_2.reshape(7, )
    return q_dot_return


def get_joint_velocities_qp_dir_inertia_specific_point_NS(fx, jacobian, manipulator, direction, alpha, l_dir, l_des, point_id):

    q_dot_1 = get_joint_velocities_qp(fx, jacobian, manipulator)
    N = (np.identity(7) - np.linalg.pinv(jacobian) @ jacobian)
    dl_dq_dir = manipulator.get_effective_inertia_point_gradient(direction, point_id)
    q_dot_2 =  -alpha * (N @ dl_dq_dir * (l_dir - l_des))
    q_dot_return = q_dot_1 + q_dot_2.reshape(7, )
    return q_dot_return



def get_joint_velocities_qp_dir_inertia_NS(fx, jacobian, manipulator, direction, alpha):

    q_dot_1 = get_joint_velocities_qp(fx, jacobian, manipulator)
    N = (np.identity(7) - np.linalg.pinv(jacobian) @ jacobian)
    # dl_dq_dir = manipulator.get_directional_inertia_gradient(direction)
    dl_dq_dir = manipulator.get_effective_inertia_gradient(direction)
    q_dot_2 = alpha * (N @ dl_dq_dir)
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

   
    q_vel = solve_qp(P, q_, G=G_, h=h_, lb=lower_bound, ub=upper_bound, solver="osqp", eps_abs=1e-2)
    return q_vel