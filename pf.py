# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 09:50:08 2019

@author: ams_user
"""
import math

import matplotlib.pyplot as plt

import numpy as np


def motion_model2(x):
    ret2=np.array([0.000,0.000,0.000,0.000,0.000])
    if x[4] == 0:
            ret2[0] = x[0] + x[2] * math.cos(x[3]) * 0.05
            ret2[1] = x[1] + x[2] * math.sin(x[3]) * 0.05
            ret2[2] = x[2]
            ret2[3] = x[3] + 0.05 * x[4]
            ret2[4] = x[4]
    else:
            ret2[0] = x[0] + (x[2] / x[4]) * (math.sin(x[3] + x[4] * 0.05) - math.sin(x[3]))
            ret2[1] = x[1] + (x[2] / x[4]) * (-math.cos(x[3] + x[4] * 0.05) + math.cos(x[3]))
            ret2[2] = x[2]
            ret2[3] = x[3] + 0.05 * x[4]
            ret2[4] = x[4]
    return ret2
    


def cul_weight(z,z_update):
    pw=np.zeros((1,5))
    pw[0,0] = (1/((2 * math.pi *0.001)**0.5)) * math.exp(-(z[0]-z_update[0])**2/(2*0.001))
    pw[0,1] = (1/((2 * math.pi *0.001)**0.5)) * math.exp(-(z[1]-z_update[1])**2/(2*0.001))
    pw[0,2] = (1/((2 * math.pi *0.001)**0.5)) * math.exp(-(z[2]-z_update[2])**2/(2*0.001))
    pw[0,3] = (1/((2 * math.pi *0.001)**0.5)) * math.exp(-(z[3]-z_update[3])**2/(2*0.001))
    pw[0,4] = (1/((2 * math.pi *0.001)**0.5)) * math.exp(-(z[4]-z_update[4])**2/(2*0.001))
    return pw