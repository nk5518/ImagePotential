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
import matplotlib.image as im
import Image as Im 

#%%
# Classes
class Hamiltonian:
    def __init__(self, IG, n_pert = 10):
        self._IG = IG
        self._npert = n_pert
        self._a, self._b = len(IG[0]), len(IG)
        self._u0 = []
        
        S0 = []
        for i in range(n_pert):
            for j in range(n_pert):
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
    
    
    def du1(self, eig, lim = 10):
        dU1 = np.zeros((self._b, self._a))
        U0 = self._u0
        ind0 = U0.index(eig)
        if ind0 - 10 <= 0:
            s = 0
        else:
            s = ind0 - 10
        friend = np.arange(start = s, stop = ind0 + 10)
        
        if type(eig) == SingleState:
            for i in friend:
                if i != ind0:
                    if type(U0[i]) == SingleState:
                        over = overlap(U0[i].get_u(), \
                                       self._IG, eig.get_u())
                        denom = eig.get_E('0') - U0[i].get_E('0')
                        dU1 += (over/denom) * U0[i].get_u()
                        
                    elif type(U0[i]) == DegenerateSet:
                        for j in range(U0[i].get_size()):
                            over = overlap(U0[i].get_u(j), \
                                       self._IG, eig.get_u())
                            denom = eig.get_E('0') - \
                                    U0[i].get_E('0', '0')
                            dU1 += (over/denom) * U0[i].get_u(j)
                            
            eig.set_du1(dU1)

            
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
        
        self._N = len(self._uD0)
        
    
    def __repr__(self):
        E = sigfig(self._E)
        return 'Deg(%s, %s)' % (E, self._N)
    
    
    def get_size(self):
        return len(self._uD0)
    
    
    def get_u(self, ind):
        return self._uD0[ind]
    
    
    def get_E(self, degree, ind):
        if degree == '0':
            return self._E
        
        if degree == '1':
            return self._dE1[ind]
        
    
    def get_R(self):
        '''
        return the resolution of the degenerate set

        Returns
        -------
        _R
            The resolution of the degenerate set. Can take 
            three possible values: '1', '2' and '0'.
                '1' - degeneracy was lifted by 1st order 
                      correction
                '2' - degeneracy was lifted by 2nd order 
                      correction
                '0' - degeneracy wasn't lifted by 1st nor
                      2nd order correction

        '''
        return self._R


    def W1(self):
        W1 = np.ones((self._N, self._N))
        for i in range(self._N):
            for j in range(self._N):
                W1[i][j] = overlap(self._uD0[i], self._IG, \
                                  self._uD0[j])
                    
        eigvec = np.linalg.eig(W1)[1]
        a, b = len(self._IG[0]), len(self._IG)
        self._dE1 = np.ones(len(self._uD0))
         
        for i in range(len(eigvec)):
            U = np.zeros((b, a))
            for j in range(len(eigvec)):
                U += eigvec[j][i] * self._uD0[j]
            
            self._dE1[i] = overlap(U, self._IG, U)
        
        l_bound = (0.01 / max((a, b)))**2 
        repeat = False
        for i in range(len(eigvec) - 1):
            for j in range(len(eigvec) - (i + 1)):
                if abs(self._dE1[i] - self._dE1[i + j + 1]) \
                    < l_bound:
                        repeat = True 
        
        if not repeat:
            self._uDnew = eigvec
            self._R = '1'
            
        else:
            self._R = '2'
            

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
        self._dE1 = overlap(self._u, IG, self._u)
        
    
    def __repr__(self):
        return 'Sin(%s)' % (sigfig(self._E))
    
    
    def get_u(self):
        return self._u
    
    
    def get_E(self, degree):
        if degree == '0':
            return self._E
        elif degree == '1':
            return self._dE1
        
    
    def set_du1(self, du1):
        self._du1 = du1
    
    
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
def W1(D, IG):
    N, a, b = D.get_size(), len(IG[0]), len(IG)
    W1 = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            W1[i][j] = overlap(D.get_u(i), IG, D.get_u(j))
                    
    eigvec = np.linalg.eig(W1)[1]
    dE1 = np.ones(D.get_size())
         
    for i in range(len(eigvec)):
        U = np.zeros((b, a))
        for j in range(len(eigvec)):
            U += eigvec[j][i] * D.get_u(j)
            
        dE1[i] = overlap(U, IG, U)
    
    l_bound = (0.01 / max((a, b)))**2
    print(l_bound)
        
    repeat = False
    for i in range(len(eigvec) - 1):
        for j in range(len(eigvec) - (i + 1)):
            if abs(dE1[i] - dE1[i + j + 1]) < l_bound:
                repeat = True 
   
    return repeat, dE1


#%%

img = im.imread('lenna.jpg')
img_g = Im.colour_drain(img)
'''
X, Y = np.linspace(0, 100, num = 100), np.linspace(0, 100, num = 100) 
x, y = np.meshgrid(X, Y)
img_g = np.exp(-(x**2 + (y**2)/10))
'''
e11 = H0(img_g, (1, 1))[0]
e21, e12 = H0(img_g, (2, 1))[0], H0(img_g, (1, 2))[0]
ds = [(e12, (1, 2)), (e21, (2, 1))]

SS = SingleState([(e11, (1, 1))], img_g)
DS = DegenerateSet(ds, img_g)
R, EPS = W1(DS, img_g)
print(R)
print(EPS)
print('')

r, eps = DS.W1()
print(r)
print(eps)
print('')

print(SS.get_E('1'))

#%%
print((2, 0) == (2, 0))
