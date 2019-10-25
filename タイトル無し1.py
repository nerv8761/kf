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
import matplotlib.animation as animation
#from pf import motion_model2,cul_weight


def iterate_x(x_in, timestep, inputs):
    '''this function is based on the x_dot and can be nonlinear as needed'''
    ret = np.zeros(len(x_in))

    if  (x_in[4] == 0):
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
    q[2][2] = 0.001
    q[3][3] = 0.001
    q[4][4] = 0.001
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
    estimatev2=[]
    realyaw=[]
    estimateyaw=[]
    estimateyaw2=[]
    realyawr=[]
    estimateyawr=[]
    estimateyawr2=[]
    estimate3x=[]
    estimate3y=[]
    ex=[]
    ey=[]
    # create measurement noise covariance matrices
    r_imu = np.zeros([1, 1])
    r_imu[0][0] = 0.001
    r_compass = np.zeros([1, 1])
    r_compass[0][0] = 0.001
    r_encoder = np.zeros([1, 1])
    r_encoder[0][0] = 0.001
    r_gpsx = np.zeros([1, 1])
    r_gpsx[0][0] = 0.001
    r_gpsy = np.zeros([1, 1])
    r_gpsy[0][0] = 0.001
    
    
    
    ini=np.array([0.000, 0.000, 0.000, 0.000, 0.000])
    xest =np.array([[0.000],
                  [0.000],
                  [0.000],
                  [0.000],
                  [0.000]])
    
    pest = np.eye(5)
    jf = np.eye(5)
    
    
    #ガウス分布の粒子の生成
    p = 100
    x_p = np.zeros((p,5))
    pw = np.zeros((p,5))
    x_p_update = np.zeros((p,5))
    z_p_update = np.zeros((p,5))
    for i in range(0,p):
        x_p[i] = np.random.randn(5)
    
    
    xp=np.zeros((1,5))
    

    # pass all the parameters into the UKF!
    # number of state variables, process noise, initial state, initial coariance, three tuning paramters, and the iterate function
    state_estimator = UKF(5, q, ini, np.eye(5), 0.04, 0.0, 2.0,iterate_x)

    
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
            
            observation_data=np.array([[row[11]],
                                       [row[12]],
                                       [row[10]],
                                       [row[9]],
                                       [row[8]]])
            
            observation_con = np.eye(5)
            observation_con[0][0] = 0.001
            observation_con[1][1] = 0.001
            observation_con[2][2] = 0.001
            observation_con[3][3] = 0.001
            observation_con[4][4] = 0.001
    

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
            
            #ekf
            ret2 = state_estimator.motion_model(xest)
            jf = state_estimator.jacobi(ret2)
            xest,pest= state_estimator.ekf_estimation(xest,pest,observation_data,observation_con,jf,ret2)
            
            """           
            #particle filter
            for i in range(0,p):
                x_p_update[i,:] = motion_model2(x_p[i,:]) 
                z_p_update[i,:] = x_p_update[i,:]
                pw[i] = cul_weight(observation_data,z_p_update[i,:])
            
            sum1=0.000
            sum2=0.000
            sum3=0.000
            sum4=0.000
            sum5=0.000
            
            for i in range(0,p):
                sum1=sum1+pw[i,0]
                sum2=sum1+pw[i,1]
                sum3=sum1+pw[i,2]
                sum4=sum1+pw[i,3]
                sum5=sum1+pw[i,4]
                
            #normalize
            for i in range(0,p):
                pw[i,0]=pw[i,0]/sum1
                pw[i,1]=pw[i,1]/sum2
                pw[i,2]=pw[i,2]/sum3
                pw[i,3]=pw[i,3]/sum4
                pw[i,4]=pw[i,4]/sum5
            
            #resampling
            for i in range(0,p):
                u = np.random.rand(1,5)
                qt1=np.zeros((1,5))
                for j in range(0,p):
                    qt1=qt1+pw[j]
                    if np.all(qt1) > np.all(u):
                        x_p[i]=x_p_update[j]
                        break
            xest2 =np.zeros((1,5))
            for i in range(0,p):
                xest2[0,0]=xest2[0,0]+x_p[i,0]
                xest2[0,1]=xest2[0,1]+x_p[i,1]
                xest2[0,2]=xest2[0,2]+x_p[i,2]
                xest2[0,3]=xest2[0,3]+x_p[i,3]
                xest2[0,4]=xest2[0,4]+x_p[i,4]
            
            xest2[0,0] = xest2[0,0] / p
            xest2[0,1] = xest2[0,1] / p
            xest2[0,2] = xest2[0,2] / p
            xest2[0,3] = xest2[0,3] / p
            xest2[0,4] = xest2[0,4] / p
            """            
                
                
            
            
            
            
            
            print("----------------------------------------------------------")
            print("Real state: ", real_state)
            print("UKF Estimated state: ", state_estimator.get_state())
            print("EKF Estimated state: ", xest.T)
            
            
            ex.append(row[11])
            ey.append(row[12])
            realx.append(real_state[0])
            realy.append(real_state[1])
            estimatex.append(state_estimator.get_state(0))
            estimatey.append(state_estimator.get_state(1))
            realv.append(real_state[2])
            estimatev.append(state_estimator.get_state(2))
            estimate2x.append(xest[0,0])
            estimate2y.append(xest[1,0])   
            estimatev2.append(xest[2,0])
            realyaw.append(real_state[3])
            estimateyaw.append(state_estimator.get_state(3))
            estimateyaw2.append(xest[3,0])
            realyawr.append(real_state[4])
            estimateyawr.append(state_estimator.get_state(4))
            estimateyawr2.append(xest[4,0])


            
    #figl=plt.figure(4)
    #plt.subplot(411)
    plt.plot(realx,realy,label="real_state")
    plt.plot(estimatex,estimatey,label="ufk_estimator")
    plt.plot(estimate2x,estimate2y,label="efk_estimator")
    #plt.plot(ex,ey,label="estimator")
    #plt.plot(estimate3x,estimate3y,label="pk_estimator")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend(loc="best")
    #plt.xlim(0,2)
    #plt.ylim(0,2)
    """
    plt.subplot(412)
    plt.plot(realv)
    plt.plot(estimatev)
    plt.plot(estimatev2)

    plt.xlabel("sample")
    plt.ylabel("v")   
    
    plt.subplot(413)
    plt.plot(realyaw)
    plt.plot(estimateyaw)
    plt.plot(estimateyaw2)
    plt.xlabel("sample")
    plt.ylabel("yaw")    
    
    plt.subplot(414)
    plt.plot(realyawr)
    plt.plot(estimateyawr)
    plt.plot(estimateyawr2)
    plt.xlabel("sample")
    plt.ylabel("yawr")
    """
    sigma1 = 0
    sigma2 = 0
    sigma3 = 0
    sigma4 = 0
    sigma5=0
    sigma6=0
    sigma7=0
    sigma8=0
    sigma9=0
    sigma10=0
    for i in range(0,len(realx)):
        sigma1 += (realx[i]-estimatex[i])**2
        sigma2 += (realy[i]-estimatey[i])**2
        sigma3 += (realx[i]-estimate2x[i])**2
        sigma4 += (realx[i]-estimate2y[i])**2
        sigma5 += (realx[i]-estimatev[i])**2
        sigma6 += (realy[i]-estimatev2[i])**2
        sigma7 += (realx[i]-estimateyaw[i])**2
        sigma8 += (realx[i]-estimateyaw2[i])**2
        sigma9 += (realx[i]-estimateyawr[i])**2
        sigma10+= (realy[i]-estimateyawr2[i])**2

    REME_x=math.sqrt(sigma1/len(realx))
    REME_y=math.sqrt(sigma2/len(realx))
    REME_x2=math. sqrt(sigma3/len(realx))
    REME_y2=math. sqrt(sigma4/len(realx))
    REME_v1=math.sqrt(sigma5/len(realx))
    REME_v2=math.sqrt(sigma6/len(realx))
    REME_y1=math. sqrt(sigma7/len(realx))
    REME_y2=math. sqrt(sigma8/len(realx))
    REME_yr1=math.sqrt(sigma9/len(realx))
    REME_yr2=math.sqrt(sigma10/len(realx))
    
    print("RMSE(ukf)_x=",REME_x,"RMSE(ekf)_x=",REME_x2)
    print("RMSE(ukf)_y=",REME_y,"RMSE(ekf)_y=",REME_y2)
    print("RMSE(ukf)_v=",REME_v1,"RMSE(ekf)_v=",REME_v2)
    print("RMSE(ukf)_yaw=",REME_y1,"RMSE(ekf)_yaw=",REME_y2)
    print("RMSE(ukf)_yawr=",REME_yr1,"RMSE(ekf)_yawr=",REME_yr2)
    
    """
    fig =[]
    plt.xlabel("x")
    plt.ylabel("y")

    plt.xlim(0,2)
    plt.ylim(0,2)
    def update_points(num):
        
        point_ani.set_data(x[num], y[num])

        return point_ani,

    x = realx
    y = realy
    

    
    fig = plt.figure(tight_layout=True)
    plt.plot(x,y)

    point_ani, = plt.plot(x[0], y[0], "ro")

    plt.grid(ls="--")
    # 开始制作动画
    ani = animation.FuncAnimation(fig, update_points, np.arange(0, 200), interval=100, blit=True)
    
    ani.save('real.gif', writer='imagemagick', fps=10)
    plt.show()
    """
 
"""
    plt.plot(realx,realy)
    plt.plot(estimatex,estimatey)
    plt.show()
"""


    
if __name__ == "__main__":
    main()
