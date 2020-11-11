#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@lms.py author: jackandriani
@moon.py author: Dr. Bui
"""
#CSC/ECE/DA 427/527
#Fall 2020

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np

eta = 0.001
num_points = 1000
dist = 0
radius = 10
width = 6
maxbeta = 50
b = 0
weightnt = np.array([[b,0]])
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

ierror = 0

for beta in range(maxbeta):
    error = []
    for i in np.random.permutation(range(2*num_points)):
        xd = np.array([[x[i], y[i]]])
        xn = xd.T  
        vd = weightt.T@xn
        
        def sgn(v):
            if v > 0:
                qrep = 1
            else:
                qrep = -1
            return(qrep)
        
        yn = sgn(vd)
        dn = d[i]
        
        en = dn - ((weightt.T)@xn)
        nweightt = weightt + (eta*xn*en)
        
        weightt = nweightt
        error.append(dn-yn)
        
        if dn-yn != 0:
            ierror = ierror + 1
    mse.append(np.mean(np.square(error)))

plt.figure(2)    
plt.plot(mse)
plt.xlabel('Number of epochs')
plt.ylabel('MSE')

x = np.linspace(-20,30,500)
y = -(b + x*(float(weightt[0]))/(float(weightt[1])))

plt.figure(1)
plt.scatter(x1,y1,c='#F80102',marker="+")
plt.scatter(x2,y2,c='#0172D8',marker="+")
plt.plot(x, y)

plt.title('Classification using LMS with distance = {}, radius = 10, and width = 6'.format(dist))
plt.xlabel('x')
plt.ylabel('y')

w = np.array([[0,0]])
eta = 0.001
vk = 1
a = 0.99
it = 5000
sige = 0.02
sigx = 0.995
MSE = []
XN = []

    
for i in range(it):
    xn = 0.001 + ((eta*0.001/2) * sigx) + sigx * ((vk**2) - ((eta*0.001)/2)) * (1-(eta*sigx)) ** (2*i)
    XN.append(xn)
    
for i in range(1,it):
    x = (a * XN[i-1]) * w
    en = (XN[i]) - (x*XN[i-1])
    w = w + (eta*XN[i]*en)
    MSE.append(np.mean(np.square(en)))
    
plt.figure(3)
plt.plot(XN,'--')
plt.plot(MSE)
plt.title('Learning-Rate Parameter n = {}'.format(eta))
plt.xlabel('Number of Iterations')
plt.ylabel('Mean-Square Error')