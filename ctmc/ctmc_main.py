import numpy as np
from patient import Patient

from ctmc_expectations import Eigen_Nij_all_times, Eigen_TauI_all_times


def EM_step(Q, patients, globalParams, pi0):
    num_state = globalParams['numStates']
    for n in range(N):
        patient = patients[n]
        chain = patient.O
        Nij = Eigen_Nij_all_times(Q, patient, chain,globalParams,pi0)
        TauI = Eigen_TauI_all_times(Q, patient,chain, globalParams,pi0)
    for i in range(num_state):
        for j in range(num_state):
            if i!=j:
                Q[i,j] = Nij[i,j]/TauI[i]
    for i in range(num_state):
        Q[i,i] = 0
        Q[i,i] = -np.sum(Q[i,:])
    return Q


if __name__=='__main__':
    Q = np.array([[-1., 0.3, 0.7], [1.,-2., 1.], [1, 3, -4]])
    Q_true = Q
    pi0 = np.array([0.5,0.25, 0.25])

    globalParams = {}

    rate1 = np.random.uniform(0,2)
    rate2 = np.random.uniform(0,2)
    rate3 = np.random.uniform(0,2)
    Q = np.array([[-rate1, rate1/2, rate1/2], [rate2/3,-rate2, 2*rate2/3], [2*rate3/4,2*rate3/4, -rate3]])
    globalParams['Q'] = Q
    globalParams['numStates'] = 3
    T_max = 5000

    N=2

    patients = []
    for n in range(N):
        patient = Patient()
        patient.initialize_randomly(Q_true,globalParams,T_max,pi0)
        patients.append(patient)

    for i in range(50):
        globalParams['Q'] = EM_step(globalParams['Q'],patients,globalParams,pi0)
        print (Q_true)
        print (globalParams['Q'])
