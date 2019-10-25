# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 10:50:16 2019

@author: ams_user
"""
from ukf import UKF
import csv
import numpy as np
import math
import matplotlib.pyplot as plt









def iterate_x(x_in, timestep, inputs):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    ret = np.zeros(len(x_in))
    if  x_in[4] == 0:
        
        ret[0] = x_in[0] + x_in[2] * math.cos(x_in[3]) * timestep
        ret[1] = x_in[1] + x_in[2] * math.sin(x_in[3]) * timestep
        ret[2] = x_in[2]
        ret[3] = x_in[3] + timestep * x_in[4]
        ret[4] = x_in[4]
    else:
        ret[0] = x_in[0] + (x_in[2] / x_in[4]) * (math.sin(x_in[3] + x_in[4] * timestep) - math.sin(x_in[3]))
        ret[1] = x_in[1] + (x_in[2] / x_in[4]) * (-math.cos(x_in[3] + x_in[4] * timestep) + math.cos(x_in[3]))
        ret[2] = x_in[2]
        ret[3] = x_in[3] + timestep * x_in[4]
        ret[4] = x_in[4]
    return ret


def main():
    np.set_printoptions(precision=3)

    # Process Noise
    q = np.eye(5)
    q[0][0] = 0.001
    q[1][1] = 0.001
    q[2][2] = 0.004
    q[3][3] = 0.025
    q[4][4] = 0.025
   # q[5][5] = 0.0025
    realx = []
    realy = []
    realv = []
    realtheta = []
    realw = []
    estimatex = []
    estimatey =[]
    estimatev = []
    estimatetheta = []
    estimatew = []
    estimate2y=[]
    estimate2x=[]

    # create measurement noise covariance matrices
    r_imu = np.zeros([1, 1])
    r_imu[0][0] = 0.01

    
    r_compass = np.zeros([1, 1])
    r_compass[0][0] = 0.02


    r_encoder = np.zeros([1, 1])
    r_encoder[0][0] = 0.001
    
    r_gpsx = np.zeros([1, 1])
    r_gpsx[0][0] = 0.1
    r_gpsy = np.zeros([1, 1])
    r_gpsy[0][0] = 0.1
    
    ini=np.array([0, 0, 0.3, 0, 0.3])

    # pass all the parameters into the UKF!
    # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
    state_estimator = UKF(5, q, ini, np.eye(5), 0.04, 0.0, 2.0, iterate_x, r_imu, r_compass, r_encoder, r_gpsx, r_gpsy)

    xest = np.zeros((5, 1))
    pest = np.eye(5)
    jf=[]
    
    with open('example.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)

        last_time = 0
        # read data
        for row in reader:
            row = [float(x) for x in row]

            cur_time = row[0]
            d_time = cur_time - last_time
            real_state = np.array([row[i] for i in [5, 6, 3, 4, 2]])

            # create an array for the data from each sensor
            compass_hdg = row[9]
            compass_data = np.array([compass_hdg])

            encoder_vel = row[10]
            encoder_data = np.array([encoder_vel])
            
            gps_x = row[11]
            gpsx_data = np.array([gps_x])
            
            gps_y = row[12]
            gpsy_data = np.array([gps_y])

           
            imu_yaw_rate = row[8]
            imu_data = np.array([imu_yaw_rate])

            last_time = cur_time
            
            observation_data=np.array([row[11],
                                      row[12],
                                      row[10],
                                      row[9],
                                      row[8]])
            observation_datac = np.array([0.1,
                                          0.1,
                                          0.001,
                                          0.02,
                                          0.01])
    

            # prediction is pretty simple
            state_estimator.predict(d_time)

            # updating isn't bad either
            # remember that the updated states should be zero-indexed
            # the states should also be in the order of the noise and data matrices
            state_estimator.update([4], imu_data, r_imu)
            state_estimator.update([3], compass_data, r_compass)
            state_estimator.update([2], encoder_data, r_encoder)
            state_estimator.update([1], gpsy_data, r_gpsy)
            state_estimator.update([0], gpsx_data, r_gpsx)
            jf = state_estimator.jacobi(xest)
 
            xest,pest= state_estimator.ekf_estimation(xest,pest,observation_data,observation_datac,jf)

            
            print("--------------------------------------------------------")
            print("Real state: ", real_state)
            print("Estimated state: ", state_estimator.get_state())
            print("Difference: ", real_state - state_estimator.get_state())
            print("Estimated state2: ", xest)




            realx.append(real_state[0])
            realy.append(real_state[1])
            estimatex.append(state_estimator.get_state(0))
            estimatey.append(state_estimator.get_state(1))
            realv.append(real_state[2])
            estimatev.append(state_estimator.get_state(2))
            estimate2x.append(xest[0])
            estimate2y.append(xest[1])   
    
    figl=plt.figure(2)
    plt.subplot(211)
    plt.plot(estimatex,estimatey,"-b",label="ufk_estimator")
    plt.plot(realx,realy,"-r",label="real_position")
    plt.plot(estimate2x,estimate2y,"-g",label="efk_estimator")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    
    plt.subplot(212)
    plt.plot(realv,"-r",label="real_v")
    plt.plot(estimatev,"-b",label="estimator_v")
    plt.legend(loc="best")

"""
    plt.plot(realx,realy)
    plt.plot(estimatex,estimatey)
    plt.show()
"""
    
if __name__ == "__main__":
    main()
