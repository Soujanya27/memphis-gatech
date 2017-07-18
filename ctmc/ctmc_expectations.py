import numpy as np
from scipy.linalg import expm


def Eigen_Nij_time_interval(Q, I_matrix, t, Psi_eigen):
    num_state = np.shape(Q)[0]
    Nij_mat = np.zeros((num_state, num_state))
    F = I_matrix / expm(Q * t)
    D, U = np.linalg.eig(Q)
    D = np.diag(D)
    V = np.linalg.inv(U)
    B = np.dot(np.dot(np.transpose(U), F), np.transpose(V))
    for i in range(num_state):
        for j in range(num_state):
            Aij = np.multiply(np.outer(V[:, i], U[j, :]), Psi_eigen)
            temp = np.multiply(Aij, B)
            if i != j:
                Nij_mat[i, j] = Q[i, j] * np.sum(temp)
    return Nij_mat


def Eigen_TauI_time_interval(Q, I_matrix, t, Psi_eigen):
    num_state = np.shape(Q)[0]
    TauI = np.zeros(num_state)
    F = I_matrix / expm(Q * t)
    D, U = np.linalg.eig(Q)
    D = np.diag(D)
    V = np.linalg.inv(U)
    B = np.dot(np.dot(np.transpose(U), F), np.transpose(V))
    for i in range(num_state):
        Ai = np.multiply(np.outer(V[:, i], U[i, :]), Psi_eigen)
        temp = np.multiply(Ai, B)
        TauI[i] = np.sum(temp)
    return TauI


def calculate_Psi_eigen(Q, t):
    num_state = np.shape(Q)[0]
    D, U = np.linalg.eig(Q)
    Psi_eigen = np.zeros((num_state, num_state))
    for p in range(num_state):
        for q in range(num_state):
            if D[p] == D[q]:
                Psi_eigen[p, q] = t * np.exp(t * D[p])
            else:
                Psi_eigen[p, q] = (np.exp(t * D[p]) - np.exp(t * D[q])) / (D[p] - D[q])
    return Psi_eigen


def Eigen_Nij_all_times(Q, patient, chain, globalParams, pi):
    """
    @param T: observation times
    :param chain:
    """
    O = patient.O
    T = patient.T_obs
    num_state = np.shape(Q)[0]
    Nij_mat = np.zeros((num_state, num_state))
    for i in range(1, T):
        t = patient.observation_times[i] - patient.observation_times[i - 1]
        I_matrix = np.zeros((num_state, num_state))
        I_matrix[int(chain[i-1]), int(chain[i-1])] = 1
        Psi_eigen = calculate_Psi_eigen(Q, t)
        Nij_mat += Eigen_Nij_time_interval(Q, I_matrix, t, Psi_eigen)
    return Nij_mat


def Eigen_TauI_all_times(Q, patient, chain, globalParams, pi):
    O = patient.O
    T = patient.T_obs
    num_state = np.shape(Q)[0]
    TauI = np.zeros(num_state)
    for i in range(1, T):
        t = patient.observation_times[i] - patient.observation_times[i - 1]
        I_matrix = np.zeros((num_state, num_state))
        I_matrix[int(chain[i-1]), int(chain[i-1])] = 1
        Psi_eigen = calculate_Psi_eigen(Q, t)
        TauI += Eigen_TauI_time_interval(Q, I_matrix, t, Psi_eigen)
    return TauI
