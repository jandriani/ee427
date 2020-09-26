#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 10:06:24 2020

@main.py author: jackandriani
@moon.py author: Dr. Bui
"""
#CSC/ECE/DA 427/527
#Fall 2020

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np

num_points = 1000
dist = 1
radius = 10
width = 6
maxbeta = 50
b = 1
weightnt = np.array([[b,0,0]])
weightt = weightnt.T
n = np.linspace(0.1, 0.00001,2*num_points)
mse = []

def moon(num_points,distance,radius,width):
 points = num_points
 x1 = [0 for _ in range(points)]
 y1 = [0 for _ in range(points)]
 x2 = [0 for _ in range(points)]
 y2 = [0 for _ in range(points)]
 d1 = [0 for _ in range(points)]
 d2 = [0 for _ in range(points)]   
 for i in range(points):
    d = distance
    r = radius
    w = width
    a = random()*math.pi
    x1[i] = math.sqrt(random()) * math.cos(a)*(w/2) + ((-(r+w/2) if(random() < 0.5) else (r+w/2)) * math.cos(a))
    y1[i] = math.sqrt(random()) * math.sin(a)*(w) + (r * math.sin(a)) + d
    d1[i] = 1
  
    a = random()*math.pi + math.pi
    x2[i] = (r+w/2) + math.sqrt(random()) * math.cos(a)*(w/2) + ((-(r+w/2)) if(random() < 0.5) else (r+w/2)) * math.cos(a)
    y2[i] = -(math.sqrt(random()) * math.sin(a)*(-w) + (-r * math.sin(a))) - d
    d2[i] = -1
 return ([x1,x2,y1,y2,d1,d2])
    
moons = moon(num_points, dist, radius, width)
x1 = moons[0]
x2 = moons[1]
y1 = moons[2]
y2 = moons[3]
d1 = moons[4]
d2 = moons[5]
x = [*x1,*x2]
y = [*y1,*y2]
d = [*d1,*d2]

xmin = min(x)-3
xmax = max(x)+3
ymin = min(y)
ymax = max(y)

y1min = min(y1)
minpos = y1.index(min(y1))
x1min = x1[minpos]

y2max = max(y2)
maxpos = y2.index(max(y2))
x2max = x2[maxpos]

plt.scatter(x1,y1,c='#F80102',marker="+")
plt.scatter(x2,y2,c='#0172D8',marker="+")
plt.plot((xmin, xmax),(y1min, y2max))
plt.title('Classification using perceptron with distance = 1, radius = 10, and width = 6')
plt.xlabel('x')
plt.ylabel('y')

ierror = 0

for beta in range(maxbeta):
    error = []
    for i in np.random.permutation(range(2*num_points)):
        xd = np.array([[1, x[i], y[i]]])
        xn = xd.T  
        vd = weightt.T@xn
        v = float(vd[0])
        def sgn(v):
            if v > 0:
                qrep = 1
            else:
                qrep = -1
            return(qrep)
        yn = sgn(v)
        dn = d[i]
        nweightt = weightt + (n[i]*(dn-yn)*xn)
        weightt = nweightt
        error.append(dn-yn)
        if dn-yn != 0:
            ierror = ierror + 1
    mse.append(np.mean(np.square(error)))
plt.plot(mse)
plt.xlabel('Number of epochs')
plt.ylabel('MSE')

errorrate = (ierror/(maxbeta*(2*num_points)))*100
