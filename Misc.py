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

#%%
# Class

class Cell:
    def __init__(self, height, pos, a, E):
        self._height = height 
        self._pos = pos
        self._E = E
        self._a = a 
        self._f = f_edge(self._pos[0] - self._a/2, self._height, self._E)
        self._b = b_edge(self._pos[0] + self._a/2, self._height, self._E)
    
    
    def __repr__(self):
        return '(%s, %s)'%(self._height, self._pos)
    
    
    def set_E(self, E):
        self._E = E 
        self._f = f_edge(self._pos[0] - self._a/2, self._height, self._E)
        self._b = b_edge(self._pos[0] + self._a/2, self._height, self._E)
        
    
    def set_height(self, height):
        self._height = height 
        self._f = f_edge(self._pos[0] - self._a/2, self._height, self._E)
        self._b = b_edge(self._pos[0] + self._a/2, self._height, self._E)

    
    def set_AB(self, AB):
        self._AB = AB
        
    
    def get_E(self):
        return self._E
    
    
    def get_f(self):
        return self._f
    
        
    def get_b(self):
        return self._b
    
    
    def get_AB(self):
        return self._AB
    
    
class Potential:
    def __init__(self, GS, a, height_matrix, E):
        self._GS = GS  #GS is the grid size
        self._a = a  #a is the cell size 
        self._E = E
        
        P = []
        for r in range(GS[0]):
            R = []
            for c in range(GS[1]):
                a_ij = Cell(H[r][c], ((c + 0.5) * a, \
                         (GS[0] - r - 0.5) * a), a, E)
                R.append(a_ij)
            P.append(R)
        self._P = P
        self._M0 = b_edge(0, 0, E)
        
    def __repr__(self):
        return '%s'%(self._P)


    def change_height(self, ind, new_height):
        self._P[ind[0]][ind[1]].set_height(new_height) 
        
        
    def change_energy(self, new_energy):
        self._E = new_energy 
        for i in range(len(self._P)):
            for j in range(len(self._P[0])):
                self._P[i][j].set_E(self._E)
        self._M0 = b_edge(0, 0, self._E)
        
    
    def get_cell(self, ind):
        return self._P[ind[0]][ind[1]]
    
    
    def back_prop(self, R):
        chain = []
        for i in range(len(R)):
            if i == 0:
                chain.append((self._M0, R[0].get_f()))
            elif i == len(R) - 1:
                chain.append(R[len(R) - 1].get_b())
            else:
                chain.append((R[i-1].get_b(), R[i].get_f()))
        W = len(R) * self._a
        vf = ex(W * root(-self._E)) * np.array([1, root(-self._E)])
        
        v = vf
        for j in [len(R) - 1 - x for x in range(len(R))]:
            if j == len(R) - 1:
                v = np.matmul(chain[j], v)
                R[j].set_AB(v)
            else:
                v = np.matmul(np.matmul(chain[j][0], chain[j][1]), v)
                R[j].set_AB(v)
        
        if R[0].get_AB()[0] == 0:
            raise Exception('A0/Af equals 0')
        else:
            Af = 1 / R[0].get_AB()[0]
        
        for i in range(len(R)):
            R[i].set_AB(Af * R[i].get_AB())
            
        
class Simulation(Potential):
    def __init__(self, E, GS, a, height_matrix):
        self._E = E
        self._P = Potential.__init__(GS, a, height_matrix, self._E)
    
    

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

#%%
