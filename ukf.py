import sympy
import numpy as np
import scipy.linalg
from copy import deepcopy
from threading import Lock
import math
import matplotlib.pyplot as plt




class UKFException(Exception):
    """Raise for errors in the UKF, usually due to bad inputs"""


class UKF:
    def __init__(self, num_states, process_noise, initial_state, initial_covar, alpha, k, beta, iterate_function):
        """
        Initializes the unscented kalman filter
        :param num_states: int, the size of the state
        :param process_noise: the process noise covariance per unit time, should be num_states x num_states
        :param initial_state: initial values for the states, should be num_states x 1
        :param initial_covar: initial covariance matrix, should be num_states x num_states, typically large and diagonal
        :param alpha: UKF tuning parameter, determines spread of sigma points, typically a small positive value
        :param k: UKF tuning parameter, typically 0 or 3 - num_states
        :param beta: UKF tuning parameter, beta = 2 is ideal for gaussian distributions
        :param iterate_function: function that predicts the next state
                    takes in a num_states x 1 state and a float timestep
                    returns a num_states x 1 state
        """
        self.n_dim = int(num_states)
        self.n_sig = 1 + num_states * 2
        self.q = process_noise
        self.x = initial_state
        self.p = initial_covar
        self.beta = beta
        self.alpha = alpha
        self.k = k
        self.iterate = iterate_function
        self.lambd = pow(self.alpha, 2) * (self.n_dim + self.k) - self.n_dim
        self.covar_weights = np.zeros(self.n_sig)
        self.mean_weights = np.zeros(self.n_sig)
        self.covar_weights[0] = (self.lambd / (self.n_dim + self.lambd)) + (1 - pow(self.alpha, 2) + self.beta)
        self.mean_weights[0] = (self.lambd / (self.n_dim + self.lambd))
       
        for i in range(1, self.n_sig):
            self.covar_weights[i] = 1 / (2*(self.n_dim + self.lambd))
            self.mean_weights[i] = 1 / (2*(self.n_dim + self.lambd))

        self.sigmas = self.__get_sigmas()
        self.lock = Lock()
        self.ini= initial_state







    
    def motion_model(self,x):
  
        ret2=np.array([[0.000],
                       [0.000],
                       [0.000],
                       [0.000],
                       [0.000]])
        if x[4,0] == 0:
            ret2[0,0] = x[0,0] + x[2,0] * math.cos(x[3,0]) * 0.05
            ret2[1,0] = x[1,0] + x[2,0] * math.sin(x[3,0]) * 0.05
            ret2[2,0] = x[2,0]
            ret2[3,0] = x[3,0] + 0.05 * x[4,0]
            ret2[4,0] = x[4,0]
        else:
            ret2[0,0] = x[0,0] + (x[2,0] / x[4,0]) * (math.sin(x[3,0] + x[4,0] * 0.05) - math.sin(x[3,0]))
            ret2[1,0] = x[1,0] + (x[2,0] / x[4,0]) * (-math.cos(x[3,0] + x[4,0] * 0.05) + math.cos(x[3,0]))
            ret2[2,0] = x[2,0]
            ret2[3,0] = x[3,0] + 0.05 * x[4,0]
            ret2[4,0] = x[4,0]

        return ret2

    def jacobi(self,ret2):
   
        
        v=ret2[2,0]
        yaw=ret2[3,0]
        yawr=ret2[4,0]
        if yawr==0:
            jf=np.array([[1.000,0.000,0.05*math.cos(yaw),-0.05*v*math.sin(yaw),0],
                         [0.000,1.000,0.05*math.sin(yaw),0.05*v*math.cos(yaw),0],
                         [0.000,0.000,1.000,0.000,0.000],
                         [0.000,0.000,0.000,1.000,0.05],
                         [0.000,0.000,0.000,0.000,1.000]])            
        else:
            x13= (-math.sin(yaw) + math.sin(yaw + 0.05*yawr))/yawr
            x14= v*(-math.cos(yaw) + math.cos(yaw + 0.05*yawr))/yawr
            x15= 0.05*v*math.cos(yaw + 0.05*yawr)/yawr - v*(-math.sin(yaw) + math.sin(yaw + 0.05*yawr))/(yawr**2)
            x23= (math.cos(yaw) - math.cos(yaw + 0.05*yawr))/yawr
            x24= v*(-math.sin(yaw) + math.sin(yaw + 0.05*yawr))/yawr
            x25= 0.05*v*math.sin(yaw + 0.05*yawr)/yawr - v*(math.cos(yaw) - math.cos(yaw + 0.05*yawr))/(yawr**2)
            jf = np.array([[1, 0, x13, x14, x15],
                           [0, 1, x23, x24, x25], 
                           [0, 0, 1, 0, 0], 
                           [0, 0, 0, 1, 0.05], 
                           [0, 0, 0, 0, 1]])
      
        return jf


 
        
        
        
    def ekf_estimation(self, Xest, Pest,data,con,jf,ret2):
        
        obs_x = data
        obs_xc = con
        
        
        
        #predict
        xpred = ret2
        
        ppred = jf @ Pest @ jf.T + self.q**2
        
        #updata
        jc = np.eye(5)
        zpred= np.eye(5) @ xpred
        x_sub = obs_x - zpred
        s = jc @ ppred @ jc.T + obs_xc**2
        k = ppred @ jc.T @ np.linalg.inv(s)
        Xest =xpred + k @ x_sub
        Pest = (np.eye(len(Xest)) - k @ jc) @ ppred
        
        return Xest,Pest






    def __get_sigmas(self):
        """generates sigma points"""
        ret = np.zeros((self.n_sig, self.n_dim))

        tmp_mat = (self.n_dim + self.lambd)*self.p

        # print spr_mat
        spr_mat = scipy.linalg.sqrtm(tmp_mat)

        ret[0] = self.x
        for i in range(self.n_dim):
            ret[i+1] = self.x + spr_mat[i]
            ret[i+1+self.n_dim] = self.x - spr_mat[i]

        return ret.T








    def update(self, states, data, r_matrix):
        self.lock.acquire()
        num_states = len(states)
        # create y, sigmas of just the states that are being updated
        sigmas_split = np.split(self.sigmas, self.n_dim)
        y = np.concatenate([sigmas_split[i] for i in states])
        # create y_mean, the mean of just the states that are being updated
        x_split = np.split(self.x, self.n_dim)
        y_mean = np.concatenate([x_split[i] for i in states])
        # differences in y from y mean
        y_diff = deepcopy(y)
        x_diff = deepcopy(self.sigmas)
        for i in range(self.n_sig):
            for j in range(num_states):
                y_diff[j][i] -= y_mean[j]
            for j in range(self.n_dim):
                x_diff[j][i] -= self.x[j]
        p_yy = np.zeros((num_states, num_states))
        for i, val in enumerate(np.array_split(y_diff, self.n_sig, 1)):
            p_yy += self.covar_weights[i] * val.dot(val.T)

        # add measurement noise
        p_yy += r_matrix

        # covariance of measurement with states
        p_xy = np.zeros((self.n_dim, num_states))
        for i, val in enumerate(zip(np.array_split(y_diff, self.n_sig, 1), np.array_split(x_diff, self.n_sig, 1))):
            p_xy += self.covar_weights[i] * val[1].dot(val[0].T)

        k = np.dot(p_xy, np.linalg.inv(p_yy))

        y_actual = data

        self.x += np.dot(k, (y_actual - y_mean))
        self.p -= np.dot(k, np.dot(p_yy, k.T))
        self.sigmas = self.__get_sigmas()       
        
        self.lock.release()

    def predict(self, timestep, inputs=[]):
        """
        performs a prediction step
        :param timestep: float, amount of time since last prediction
        """

        self.lock.acquire()

        sigmas_out = np.array([self.iterate(x, timestep, inputs) for x in self.sigmas.T]).T

        x_out = np.zeros(self.n_dim)

        # for each variable in X
        for i in range(self.n_dim):
            # the mean of that variable is the sum of
            # the weighted values of that variable for each iterated sigma point
            x_out[i] = sum((self.mean_weights[j] * sigmas_out[i][j] for j in range(self.n_sig)))

        p_out = np.zeros((self.n_dim, self.n_dim))
        # for each sigma point
        for i in range(self.n_sig):
            # take the distance from the mean
            # make it a covariance by multiplying by the transpose
            # weight it using the calculated weighting factor
            # and sum
            diff = sigmas_out.T[i] - x_out
            diff = np.atleast_2d(diff)
            p_out += self.covar_weights[i] * np.dot(diff.T, diff)

        # add process noise
        p_out += timestep * self.q 

        self.sigmas = sigmas_out
        self.x = x_out
        self.p = p_out
        
        self.lock.release()


    def get_state(self, index=-1):
        """
        returns the current state (n_dim x 1), or a particular state variable (float)
        :param index: optional, if provided, the index of the returned variable
        :return:
        """
   
        if index >= 0:
            return self.x[index]
        else:
            return self.x

    def get_covar(self):
        """
        :return: current state covariance (n_dim x n_dim)
        """
        return self.p



    def set_state(self, value, index=-1):
        """
        Overrides the filter by setting one variable of the state or the whole state
        :param value: the value to put into the state (1 x 1 or n_dim x 1)
        :param index: the index at which to override the state (-1 for whole state)
        """
        with self.lock:
            if index != -1:
                self.x[index] = value
            else:
                self.x = value

    def reset(self, state, covar):
        """
        Restarts the UKF at the given state and covariance
        :param state: n_dim x 1
        :param covar: n_dim x n_dim
        """

        with self.lock:
            self.x = state
            self.p = covar
    
    