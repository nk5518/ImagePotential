# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:21:20 2020

@author: non_k
"""

#%%
''' Edit: 19/07/20 '''
#%%


# Schroedinger Equation


#%%
import numpy as np
import pylab as pl
import Image as Im 

#%%
# Classes
class Hamiltonian:
    def __init__(self, IG):
        self._IG = IG
        self._a, self._b = len(IG[0]), len(IG)
        self._u0 = {}
        
        S0 = []
        for i in range(10):
            for j in range(10):
                S0.append(((i + 1, j + 1), \
                           H0(self._IG, (i + 1, j + 1))[0]))  
        
        ref = len(S0)
        count = 0
        while len(S0) > 0:
            if count > ref + 10:
                raise RuntimeError('Sorting ran overtime')
                
            ind, obj = [0], [S0[0]]
            for i in range(len(S0) - 1):
                if abs(S0[0][1] - S0[i + 1][1]) < \
                    (0.01 / max((self._a, self._b)))**2:
                        ind.append(i + 1)
                        obj.append(S0[i + 1])
            
            if len(ind) == 1:
                self._u0[count] = H0(self._IG, obj[0][0])
                del S0[0]
            
            else: 
                self._u0[count] = DegenerateSet(obj[0][1], obj, self._IG)
                for i in ind:
                    del S0[i]
                    
            count += 1 
            
            
class DegenerateSet:
    def __init__(self, E, uD0, IG):
        self._E = E 
        self._uD0 = {}
        for i in range(len(uD0)):
            self._uD0[i] = H0(IG, uD0[i][0])                     


# Miscellaneous Functions

def H0(F, nm):
    a, b = len(F[0]), len(F)
    x, y = np.meshgrid(np.linspace(0, a, num = a), \
                       np.linspace(0, b, num = b))
    f = np.sin((np.pi / a) * nm[0] * x) * \
        np.sin((np.pi / b) * nm[1] * y)
    E = (nm[0] / a)**2 + (nm[1] / b)**2 
        
    return E, np.sqrt(4 / (a * b)) * f


def i_2D(Arr):
    a, b = len(Arr[0]), len(Arr)
    dx, dy = a / (a - 1), b / (b - 1)
    mx, my = np.ones(a), np.ones(b)
    mx[0], mx[-1], my[0], my[-1] = 1/2, 1/2, 1/2, 1/2
    mx, my = mx * dx, my * dy
    
    V = []
    for i in range(len(Arr)):
        V.append(np.dot(mx, Arr[i]))

    return np.dot(my, V)   


def expectation(bra, H, ket):
    if type(H) == int:
        H = np.ones((len(bra), len(bra[0])))
    
    integrand = np.multiply(np.conj(bra), np.multiply(H, ket))
    
    return i_2D(integrand)

        
#%%
u = H0(np.ones((50, 50)), (1, 1))[1]
print(i_2D(u))