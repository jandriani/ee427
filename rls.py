#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 17:24:14 2020

@rls.py author: jackandriani
@moon.py author: Dr. Bui
"""
#CSC/ECE/DA 427/527
#Fall 2020

from random import random
import matplotlib.pyplot as plt
import math
import numpy as np

lamda = 0
num_points = 1000
dist = 1
radius = 10
width = 6
maxbeta = 50
b = 0
I = np.identity(2)

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

def merge(d1, d2): 
    merged_list = [(d1[i], d2[i]) for i in range(0, len(d1))] 
    return merged_list
    
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

## Least Squares Classification

total = []

for i in np.random.permutation(range(2*num_points)):
    xd = np.array([[x[i], y[i], d[i]]])
    xn = xd.T 
    total.append(xn)
    
Rx = [item[0] for item in total]
Ry = [item[1] for item in total]
r = [item[2] for item in total]

Rnt = merge(Rx, Ry)
R = (np.array((Rx, Ry))).T
Rt = (np.array(Rnt)).T

actR = np.ndarray.tolist(R)   # convert to lists from numpy arrays
actRt = np.ndarray.tolist(Rt)

dotpr = Rt.dot(R)
dotpr = np.reshape(dotpr, (2,2))   # reshaping dimensions of dot product to 2x2

w = (np.linalg.inv(dotpr + (lamda * I))).dot((Rt.dot(r)))

x = np.linspace(-20,30,500)
y = -(b + x*(float(w[0]))/(float(w[1])))

plt.scatter(x1,y1,c='#F80102',marker="+")
plt.scatter(x2,y2,c='#0172D8',marker="+")
plt.title('Classification using least squares with distance = {}, radius = 10, and width = 6'.format(dist))
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x,y)
plt.axis([-20, 30, -18, 18])