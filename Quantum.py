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

#%%
# Classes
class Hamiltonian:
    def __init__(self, IG):
        self._IG = IG
        self._a, self._b = len(IG[0]), len(IG)
        self._u0 = []
        
        S0 = []
        for i in range(10):
            for j in range(10):
                t = (H0(self._IG, (i + 1, j + 1))[0], \
                     (i + 1, j + 1))
                S0.append(t)

        S0 = ascend(S0)
        d = []
        
        while len(S0) > 0:
            if len(S0) > 2:
                if abs(S0[0][0] - S0[1][0]) < \
                    (0.01 / max((self._a, self._b)))**2:
                        d.append(S0.pop(0))
                else:
                    d.append(S0.pop(0))
                    self._u0.append(d)
                    d = []
            
            elif len(S0) == 2:
                if abs(S0[0][0] - S0[1][0]) < \
                    (0.01 / max((self._a, self._b)))**2:
                        d.append(S0.pop(0))
                        d.append(S0.pop(0))
                        self._u0.append(d)
                        d = []
                else:
                    d.append(S0.pop(0))
                    self._u0.append(d)
                    self._u0.append([S0.pop(0)])
                    d = []
    
        for i in range(len(self._u0)):
            if len(self._u0[i]) == 1:
                self._u0[i] = SingleState(self._u0[i], self._IG)
            elif len(self._u0[i]) > 1:
                self._u0[i] = DegenerateSet(self._u0[i], self._IG)
     
    
    def dE1(self):
        return 'banana'
    
    
    def du1(self):
        return 'banana'
            
            
class DegenerateSet:
    def __init__(self, info, IG):
        '''
        Parameters
        ----------
        info : ndarray
            a list of tuples, whose first element is the energy, 
            and the second being the quantum numbers
        IG : ndarray
            grey-scaled image.

        Returns
        -------
        generates an object - a set of degenerate eigenstate.

        '''
        self._IG = IG
        self._E = info[0][0]
        self._uD0 = {}
        for i in range(len(info)):
            self._uD0[i] = H0(IG, info[i][1])[1]                     

    
    def __repr__(self):
        E = sigfig(self._E)
        self._N = len(self._uD0)
        return 'Deg(%s, %s)' % (E, self._N)
    
    
    def get_u(self):
        return self._uD0


    def W1(self):
        W = np.ones((self._N, self._N))
        for i in range(self._N):
            for j in range(self._N):
                W[i][j] = overlap(self._uD0[i], self._IG, \
                                  self._uD0[j])
        

class SingleState:
    def __init__(self, info, IG):
        '''
        Parameters
        ----------
        info : ndarray
            a list of tuples, whose first element is the energy, 
            and the second being the quantum numbers
        IG : ndarray
            grey-scaled image.

        Returns
        -------
        generates an object - a non-degenerate eigenstate.

        '''
        self._E = info[0][0]
        self._u = H0(IG, info[0][1])[1]
        
    
    def __repr__(self):
        return 'Sin(%s)' % (sigfig(self._E))
    
    
    def get_u(self):
        return self._u
    
    
    
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


def overlap(bra, H, ket):
    if type(H) == int:
        H = np.ones((len(bra), len(bra[0])))
    
    integrand = np.multiply(np.conj(bra), np.multiply(H, ket))
    
    return i_2D(integrand)


def ascend(o):
    o_type = [float, int, tuple]
    if type(o[0]) not in o_type:
        raise TypeError
   
    status, count = 'incomplete', 0
    
    while status == 'incomplete':
        status = 'complete'
        if count > 10 * len(o):
            raise RuntimeError('overran')
        
        for i in range(len(o) - 1): 
            if type(o[0]) in [float, int]:
                key = o[i] > o[i + 1]
            elif type(o[0]) == tuple:
                key = o[i][0] > o[i + 1][0]
                
            if key:
                status = 'incomplete'
                high, low = o[i], o[i + 1]
                o[i], o[i + 1] = low, high
            
    return o
    

def sigfig(n):   
    digit = 3 - int(np.log10(n))
    return round(n, digit)

    
#%%
IG = np.ones((100, 100))
a, b = 100, 100
u0 = {}

S0 = []
for i in range(10):
    for j in range(10):
        t = (H0(IG, (i + 1, j + 1))[0], (i + 1, j + 1))
        S0.append(t)
        
        '''
        element of S0 is a tuple such that:
            first element - energy
            second elemet - tuple of quantum number
        '''
        
S0 = ascend(S0)

u0 = []
d = []
while len(S0) > 0:
    if len(S0) > 2:
        if abs(S0[0][0] - S0[1][0]) < \
            (0.01 / max((a, b)))**2:
                d.append(S0.pop(0))
        else:
            d.append(S0.pop(0))
            u0.append(d)
            d = []
    elif len(S0) == 2:
        if abs(S0[0][0] - S0[1][0]) < \
            (0.01 / max((a, b)))**2:
                d.append(S0.pop(0))
                d.append(S0.pop(0))
                u0.append(d)
                d = []
        else:
            d.append(S0.pop(0))
            u0.append(d)
            u0.append([S0.pop(0)])
            d = []
    
for i in range(len(u0)):
    if len(u0[i]) == 1:
        u0[i] = SingleState(u0[i], IG)
    elif len(u0[i]) > 1:
        u0[i] = DegenerateSet(u0[i], IG)
    
print(u0)    
   
#%%
# reference for grouping 
E = []
for i in range(10):
    for j in range(10):
      E.append(int((i + 1)**2 + (j + 1)**2))

E = ascend(E)
new = []

d = []
while len(E) > 0:
    if len(E) != 2:
        if E[0] == E[1]:
            d.append(E.pop(0))
        else:
            d.append(E.pop(0))
            new.append(d)
            d = []
    elif len(E) == 2:
        if E[0] == E[1]:
            d.append(E.pop(0))
            d.append(E.pop(0))
            new.append(d)
            d = []
        else:
            d.append(E.pop(0))
            new.append(d)
            new.append([E.pop(0)])
            d = []
        
print(new)       

