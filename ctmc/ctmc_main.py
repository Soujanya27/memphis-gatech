import numpy as np
from patient import Patient
import csv
import os

#I suspect that the issue is one of the following:
#-

from ctmc_expectations import Eigen_Nij_all_times, Eigen_TauI_all_times


def EM_step(Q, patients, globalParams, N,pi0):
    num_state = globalParams['numStates']
    Nij = 0
    TauI = 0
    total_time = 0
#    for n in range(N):
    n=4
    patient = patients[n]
    chain = patient.O
    Nij += Eigen_Nij_all_times(Q, patient, chain,globalParams,pi0)
    TauI += Eigen_TauI_all_times(Q, patient,chain, globalParams,pi0)
    print TauI
    total_time += patient.observation_times[-1]
    for i in range(num_state):
        for j in range(num_state):
            if i!=j:
                Q[i,j] = Nij[i,j]/TauI[i]
    for i in range(num_state):
        Q[i,i] = 0
        Q[i,i] = -np.sum(Q[i,:])
    print np.sum(TauI)
    print total_time
    return Q

def preprocessing():
    patients = []
    with open('states_new.csv','rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        patient = Patient()
        patient.id = 1
        i = 0
        chain = []
        times = []
        divider = 100
        for row in reader:
            #check whether patient the same one
            if i==0:
                i+=1
                continue
            if int(row[0])==patient.id:
                chain.append(int(row[4]))
                times.append(float(row[5])/divider)
            else:
                #move to next patient
                patient.O = np.array(chain)
                patient.observation_times = np.array(times)
                patient.T_obs = len(times)
                patients.append(patient)
                patient = Patient()
                patient.id = int(row[0])
                chain = [int(row[4])]
                times = [float(row[5])/divider]
            i+=1

        #setup final patient
        patient.O = np.array(chain)
        patient.observation_times = np.array(times)
        patient.T_obs = len(times)
        patients.append(patient)
        patient = Patient()
        patient.id = int(row[0])
    return patients



if __name__=='__main__':
    np.random.seed(3)
    patients = preprocessing()
    for patient in patients:
        print patient.id
    pi0 = np.array([0.25,0.25,0.25,0.25])

    globalParams = {}
    d = 4
    Q = np.random.uniform(0,10,(d,d))
    np.fill_diagonal(Q,np.zeros(d))
    for i in range(d):
        Q[i,i] = -np.sum(Q[i,:])
    print Q
    globalParams['Q'] = Q
    globalParams['numStates'] = d
    N = len(patients)
    for i in range(5):
        globalParams['Q'] = EM_step(globalParams['Q'],patients,globalParams,N,pi0)
        print (globalParams['Q'])
