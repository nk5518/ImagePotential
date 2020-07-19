# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 15:31:38 2020

@author: non_k
"""

#%%
''' Edit: 19/07/20 '''
#%%


# Heat Equation


#%%
import numpy as np
import pylab as pl
from matplotlib.animation import FuncAnimation 

#%%
# Class
class Simulation:
    def __init__(self, I):
        Vmax = I[0][0]
        for i in range(len(I)):
            for j in range(len(I[0])):
                if max(I[0]) > Vmax:
                    Vmax = max(I[0])
        
        self._I = (255 / Vmax) * I 
        self._a, self._b  = len(I[0]), len(I)
        
        
    def U0(self, N):
        self._N = N
        self._U = {}  
        for i in range(self._N):
            for j in range(self._N):
                self._U[(i + 1, j + 1)] = Anm(self._I, \
                                              (i + 1, j + 1))
        
    
    def init(self):
        D = np.zeros((self._b, self._a))
        for i in range(self._N):
            for j in range(self._N):
                D += self._U[(i + 1, j + 1)]
                
        self._im.set_data(D)
        self._im.autoscale()
        
        return [self._im] 
    
    
    def update(self, f):
        D = np.zeros((self._b, self._a))
        for i in range(self._N):
            for j in range(self._N):
               D += self._U[(i + 1, j + 1)] * \
                    T(f, (i + 1, j + 1), (self._a, self._b)) 
        
        self._im.set_data(D)
        self._im.autoscale()
        
        return [self._im] 
    
    
    def play(self, N):
        self.U0(N)
        self._fig, self._ax = pl.subplots()
        self._im = self._ax.imshow(np.zeros((self._b, self._a)))
        
        tau = ((1 / self._a)**2 + (1 / self._b)**2) * \
              (np.pi)**2 
        t = np.linspace(0, 0.1/tau, num = 200)
        FuncAnimation(self._fig, self.update, t, self.init, \
                      interval = 50, repeat = False, blit = True)
        
        
# Static Function 
def Anm(F, nm):
    a, b, n, m = len(F[0]), len(F), nm[0], nm[1]
    X, Y = np.linspace(0, a, num = a), np.linspace(0, b, num = b)
    x, y = np.meshgrid(X, Y)
    F_ker = F * np.sin((np.pi / a) * n * x) * \
                np.sin((np.pi / b) * m * y) * (4 / (a * b))

    dx, dy = abs(X[1] - X[0]), abs(Y[1] - Y[0])    
    mx, my = np.ones(len(X)), np.ones(len(Y))
    mx[0], mx[-1], my[0], my[-1] = 1/2, 1/2, 1/2, 1/2
    mx, my = dx * mx, dy * my
    
    V = []
    for i in range(len(F_ker)):
        V.append(np.dot(mx, F_ker[i]))
        
    A = np.dot(my, V)
    
    return A * np.sin((np.pi / a) * n * x) * \
               np.sin((np.pi / b) * m * y)
    

def T(t, nm, ab):
    n, m, a, b = nm[0], nm[1], ab[0], ab[1]
    K = np.pi**2 * ((n / a)**2 + (m / b)**2)
    
    return np.exp(-K*t)
