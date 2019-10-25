# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 13:28:46 2019

@author: ams_user
"""

import numpy as np
import math




q = np.eye(5)
q[0][0] = 0.0001
q[1][1] = 0.0001
q[2][2] = 0.0004
q[3][3] = 0.0025
q[4][4] = 0.0025


print(q)