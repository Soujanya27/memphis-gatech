import numpy as np
from patient import Patient

from ctmc_expectations import Eigen_Nij_all_times, Eigen_TauI_all_times


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
    T_max = 5

    patient = Patient()
    patient.initialize_randomly(Q_true,globalParams,T_max,pi0)
    print patient.latent_trajectory
    print patient.T_latent
    print patient.observation_times
    print patient.O
