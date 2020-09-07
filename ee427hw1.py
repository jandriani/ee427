#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 09:09:43 2020

@author: jackandriani
"""

# %% 2 Input MP Neuron
x = [0,1]

def MPneuron(x):
        xs = sum(x)
        y = xs + 1
        if y >= 0:
            print(1)
        elif y < 0:
            print(0)

MPneuron(x)

# %% 5 Input MP Neuron
x = [0,1,1,0,1]

def MPneuron(x):
        xs = sum(x)
        y = xs + 1
        if y >= 0:
            print(1)
        elif y < 0:
            print(0)

MPneuron(x)

# %% n Input MP Neuron
# You can have as many inputs as you would like
x = [0,1,1,0,1,0,0]

def MPneuron(x):
        xs = sum(x)
        y = xs + 1
        if y >= 0:
            print(1)
        elif y < 0:
            print(0)

MPneuron(x)
