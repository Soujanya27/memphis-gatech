import numpy as np
from scipy.linalg import expm
from scipy.misc import logsumexp
import pandas as pd


class Patient:
    def initialize_randomly(self,Q,globalParams,T_max,pi0):
        self.generate_latent_trajectory(Q,T_max,pi0)
        self.generate_observation_times(1,T_max)
        self.get_observation_trajectory()
    
    def generate_latent_trajectory(self,Q,T_max,pi0):
        '''
        @summary: use transition rate matrix Q to generate latent trajectory
        @param Q: transition matrix
        @param pi0: start probability
        '''
        trajectory = []
        state = self.discrete_sampler(pi0)
        t = 0.0
        trajectory.append([state, t])
        U = np.random.uniform(0,1)
        covariates = []
        while t<T_max:
            #get the rate
            rate = -Q[state,state]
            covariates.append(np.array([state+1.]))
            hazard = 1.
            #get next jump time
            jump_interval = np.random.exponential(1./rate)
            increment = jump_interval*hazard
            #get new state
            #if t<T_max, get row, divide all non-diagonal elements by the rate
            p_vector = Q[state,:]/rate
            #add the rate to the diagonal element to get a 0
            p_vector[state]+=1
            #sample from this vector to get the new state
            state = self.discrete_sampler(p_vector)
            t += jump_interval
            #append this to the trajectory
            trajectory.append([state,t])
        self.latent_trajectory = np.array(trajectory)
        self.T_latent = np.shape(self.latent_trajectory)[0]

    def generate_observation_times(self,rate,T_max):
        t = 0
        T = []
        T.append(t)
        while t < T_max:
            t = t+np.random.exponential(1./rate)
            if t < T_max:
                T.append(t)
        self.observation_times = np.array(T)
        self.T_obs = len(T)

    def get_observation_trajectory(self):
        O = []
        for t in self.observation_times:
            ind = np.searchsorted(self.latent_trajectory[:,1],t)
            O.append(self.latent_trajectory[ind-1,0])
        self.O = np.array(O)
            

    def discrete_sampler(self,pi0):
        val = np.random.uniform(0,1)
        total_prob = 0
        for i in range(len(pi0)):
            total_prob += pi0[i]
            if val < total_prob:
                return i

    def get_zeta(self,t,alpha,beta, globalParams,observations_t):
        '''
            @summary: calculate zeta from the paper
            @param i,j: state from and to
            @param t: time interval length
            @param alpha: alpha vector for time t
            @param beta: beta vector for time t+1
            @param globalParams: as everywhere
            @param observations_t: all observations for time t, all types
            @return: the zeta for state i to j
            '''
        Q = globalParams['Q']
        b = self.b_s(observations_t)
        likelihood = np.dot(alpha,np.dot(expm(Q*t),beta*b))
        return expm(Q*t)*np.outer(alpha, np.transpose(b*beta))/likelihood

