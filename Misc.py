# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 21:52:24 2020

@author: non_k
"""

#%%


# Miscellaneous 


#%%
import numpy as np
import pylab as pl
from matplotlib.animation import FuncAnimation 

#%%
# Class

class Cell:
    def __init__(self, height, pos, a, E):
        self._height = height 
        self._pos = pos
        self._E = E
        self._a = a 
        self._f = f_edge(self._pos[0] - self._a/2, \
                         self._height, self._E)
        self._b = b_edge(self._pos[0] + self._a/2, \
                         self._height, self._E)
    
    
    def __repr__(self):
        return '(%s, %s)'%(self._height, self._pos)
    
    
    def set_E(self, E):
        self._E = E 
        self._f = f_edge(self._pos[0] - self._a/2, \
                         self._height, self._E)
        self._b = b_edge(self._pos[0] + self._a/2, \
                         self._height, self._E)
        
    
    def set_height(self, height):
        self._height = height 
        self._f = f_edge(self._pos[0] - self._a/2, \
                         self._height, self._E)
        self._b = b_edge(self._pos[0] + self._a/2, \
                         self._height, self._E)

    
    def set_AB(self, AB):
        self._AB = AB
        
    
    def get_pos(self):
        return self._pos
    
    
    def get_E(self):
        return self._E
    
    
    def get_height(self):
        return self._height
    
    
    def get_f(self):
        return self._f
    
        
    def get_b(self):
        return self._b
    
    
    def get_AB(self):
        return self._AB
    
    
    def wavefunc(self):
        X = np.linspace(self._pos[0] - self._a/2, \
                        self._pos[0] + self._a/2, num = 30)
        Y = np.linspace(self._pos[1] - self._a/2, \
                        self._pos[1] + self._a/2, num = 30)
        x, y = np.meshgrid(X, Y)
        wf = sol(x, self._height, self._AB, self._E) * \
             np.sin((np.pi/self._a) * y)**2
        return x, y, wf

    
class Potential:
    def __init__(self, HM, a):
        self._GS = (len(HM), len(HM[0]))
        self._a = a
        self._E = 0 
        
        P = []
        for r in range(len(HM)):
            R = []
            for c in range(len(HM[0]) + 1):
                if c == 0:
                    a_ij = Cell(0, ((c - 0.5) * a, \
                                (len(HM) - r - 0.5) * a), a, 0)
                else:
                    a_ij = Cell(HM[r][c - 1], ((c - 0.5) * a, \
                                (len(HM) - r - 0.5) * a), a, 0)
                R.append(a_ij)
            P.append(R)
        self._P = P

        
    def __repr__(self):
        P_r = []
        for i in range(len(self._P)):
            R_r = []
            for j in range(len(self._P[0])):
                if j != 0:
                    R_r.append(self._P[i][j])
            P_r.append(R_r)
        return '%s'%(P_r)


    def change_height(self, ind, new_height):
        if ind[0] < 0 or ind[1] < 0:
            raise Exception('index out of range')
        
        self._P[ind[0]][ind[1] + 1].set_height(new_height) 
        
        
    def change_energy(self, new_energy):
        self._E = new_energy
        
        for i in range(len(self._P)):
            for j in range(len(self._P[0])):
                self._P[i][j].set_E(self._E)
        
    
    def get_gridsize(self):
        return self._GS
    
    
    def get_cell(self, ind):
        if ind[0] < 0 or ind[1] < 0:
            raise Exception('index out of range')
        return self._P[ind[0]][ind[1] + 1]
    
    
    def get_E(self):
        return self._E
    
    
    def back_prop(self, r):
        R = self._P[r] 
        chain = []
        for i in [len(R) - 1 - x for x in range(len(R))]:
            if i == len(R) - 1:
                chain.append(R[i].get_b())
            else:
                chain.append((R[i].get_b(), R[i+1].get_f()))
            
        W = len(R) * self._a
        vf = ex(W * root(-self._E)) * np.array([1, root(-self._E)])
        
        v = vf
        for i in range(len(chain)):
            if i == 0:
                v = np.matmul(chain[i], v)
            else:
                v = np.matmul(np.matmul(chain[i][0], \
                              chain[i][1]), v)
            R[len(R) - 1 - i].set_AB(v)
    
        Af = 1/R[0].get_AB()[0]
    
        for i in range(len(R)):
            R[i].set_AB(R[i].get_AB() * Af)                
    
    
class Simulation:
    def __init__(self, HM, a):
        self._a = a
        self._P = Potential(HM, a)
        self._row, self._col = self._P.get_gridsize() 
        
    
    def initial(self):
        X, Y, Z = snap(self._P, 0)
        XY, Z = np.c_[X, Y], np.array(Z)
    
        self._scat.set_offsets(XY)
        self._scat.set_array(Z)
    
        return self._scat, 


    def update(self, i):
        X, Y, Z = snap(self._P, i)
        XY, Z = np.c_[X, Y], np.array(Z)
    
        self._scat.set_offsets(XY)
        self._scat.set_array(Z)
    
        return self._scat, 
    

    def play(self):
        self._fig, self._ax = pl.subplots()
        self._ax.axis([0, self._col * self._a, \
                       0, self._row * self._a])
        self._scat = self._ax.scatter([], [])
        
        Vmax = self._P.get_cell((0, 0)).get_height()
        for i in range(self._row):
            for j in range(self._col):
                if self._P.get_cell((i, j)).get_height() > Vmax:
                    Vmax = self._P.get_cell((i, j)).get_height()
        
        E = np.linspace(0, 2 * Vmax)
        FuncAnimation(self._fig, self.update, E, interval = 20, \
                      init_func = self.initial, blit = True, \
                      repeat = False)
        pl.show()
        
        
# Function 

def root(n):
    if n >= 0:
        return np.sqrt(n)
    else:
        n *= -1 
        return np.sqrt(n)*1j
    

def ex(a):
    return np.exp(a)
    

def f_edge(x, v, e):
    if v == e:
        return np.array([[1, x], [0, 1]])
    else:
        k = root(v - e)
        Ex = ex(k * x)
        return np.array([[Ex, 1 / Ex], [k * Ex, -k / Ex]])


def b_edge(x, v, e):
    return np.linalg.inv(f_edge(x, v, e))


def sol(x, v, V, e):
    if v == e:
        f = V[0] + V[1] * x
    else:
        k = root(v - e)
        f = V[0] * ex(k * x) + V[1] * ex(-k * x)
    return f * np.conj(f)


def snap(A, E, show = False):
    A.change_energy(E)
    row, col = A.get_gridsize()
    for i in range(row):
        A.back_prop(i)
    
        
    X, Y, Z = [], [], []
    for i in range(row):
        for j in range(col):
            x, y, z = A.get_cell((i, j)).wavefunc()
            for r in range(len(z)):
                for c in range(len(z[0])):
                    X.append(x[r][c])
                    Y.append(y[r][c])
                    Z.append(np.real(z[r][c]))
                    
    if show: 
        fig, ax = pl.subplots()
        ax.scatter(X, Y, s = 3, c = np.real(Z))
        pl.show()
    
    return X, Y, Z    
    
