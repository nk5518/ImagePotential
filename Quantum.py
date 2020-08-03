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
    def __init__(self, IG, gen = 10):
        self._IG = IG
        self._a, self._b = len(IG[0]), len(IG)
        self._u0 = []
        
        S0 = []
        for i in range(gen):
            for j in range(gen):
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
    
    
    def get_size(self):
        return len(self._u0)
    
    
    def du1(self, ind0, lim = 10):
        U0 = self._u0
        eig = U0[ind0]
        if ind0 - lim <= 0:
            s = 0
        else:
            s = ind0 - lim
            
        if ind0 + lim >= len(self._u0) - 1:
            e = len(self._u0) - 1
        else:
            e = ind0 + lim 
        rad = np.arange(start = s, stop = e)
        
        if type(eig) == SingleState:
            eig.set_du1(self.out_1(eig.get_u(), eig.get_E('0'), \
                                   ind0, rad))
        
        elif type(eig) == DegenerateSet:
            for i in range(eig.get_size()):
                ui, ei = eig.get_u(i), eig.get_E('0', '0')
                dU1_out = self.out_1(ui, ei, ind0, rad)
                
                eig.set_dE2(i, overlap(ui, self._IG, dU1_out))
                dU1_in = np.zeros((self._b, self._a))
                for j in range(eig.get_size()):
                    if j != i:
                        ov = overlap(eig.get_u(j), self._IG, \
                                     dU1_out)
                        de = eig.get_E('1', i) - eig.get_E('1', j)
                        dU1_in += (ov / de) * eig.get_u(j)
                eig.set_du1(i, dU1_in + dU1_out)  
                
                        
    def out_1(self, u_ref, e_ref, ind, radius):
        '''
        contribution to 1st order correction from outside of 
        the degenerate subspace.

        Parameters
        ----------
        u_ref : ndarray
            the wavefunction whose correction is being calculated
        e_ref : float
            the zero-order energy associated with u_ref
        ind0 : int
            index of the objects, SingleState and DegenerateSet,
            in _u0 of the Hamiltonian object
        radius: ndarray
            an array of indices of eigenstates to be included 
            in the correction
            
        Returns
        -------
        dU1_out: ndarray
            the contribution, from outside of the degenerate 
            subspace, to the 1st order wavefunction correction 

        '''      
        U0 = self._u0        
        dU1_out = np.zeros((self._b, self._a))
        for i in radius:
            if i != ind:
                if type(U0[i]) == SingleState:
                    over = overlap(U0[i].get_u(), self._IG, u_ref)
                    denom = e_ref - U0[i].get_E('0')
                    dU1_out += (over/denom) * U0[i].get_u()
                            
                elif type(U0[i]) == DegenerateSet:
                    for j in range(U0[i].get_size()):
                        over = overlap(U0[i].get_u(j), \
                                       self._IG, u_ref)
                        denom = e_ref - U0[i].get_E('0', '0')
                        dU1_out += (over/denom) * U0[i].get_u(j)
       
        return dU1_out
        
        
    def display_u(self, ind, lim = 10):
        self.du1(ind, lim)
        L = 0.8/self._u0[ind].lamb_lim()
        if type(self._u0[ind]) == SingleState:
            fig, ax = pl.subplots()
            psi = self._u0[ind].U(L)
            pl.imshow(np.conj(psi) * psi)
            fig.text(0.71, 0.91, '%s' % (ind), \
                     bbox = dict(edgecolor = 'black', \
                                 alpha = 0.5))
        
        elif type(self._u0[ind]) == DegenerateSet:
            for i in range(self._u0[ind].get_size()):
                fig, ax = pl.subplots()
                psi = self._u0[ind].U(i, L)
                pl.imshow(np.conj(psi) * psi)
                fig.text(0.71, 0.91, '%s; %s' % (ind, i), \
                         bbox = dict(edgecolor = 'black', \
                                     alpha = 0.5))
    
    
    def energy_split(self, ind, lim = 10):
        ind = ascend(ind)
        for i in range(len(ind)):
            self.du1(i, lim)
        
        fig, ax = pl.subplots()
        if type(self._u0[ind[-1]]) == SingleState:
            Vmax = self._u0[ind[-1]].get_E('0')
        elif type(self._u0[ind[-1]]) == DegenerateSet:
            Vmax = self._u0[ind[-1]].get_E('0', '0')
        ax.set_ylim(0, 1.5 * Vmax)
        
        l = []
        for i in range(len(ind)):
            l.append(self._u0[i].lamb_lim())
        l_max = max(l)
        L = np.linspace(0, 0.8/l_max, num = 100)
        
        for i in range(len(ind)):
            if type(self._u0[i]) == SingleState:
                mag = -int(np.log10(self._u0[i].E(0)))
                ax.plot(L, self._u0[i].E(L) * 10**mag, \
                        color = 'black')
                en = self._u0[i].get_E('0')
                ax.annotate('%s' % (i), (0, en), \
                            (0, en + 5e-2 * Vmax))
                ax.set_xlabel(r'$\lambda$')
                ax.set_ylabel('E / $10^{%s}$' % (-mag))
                
            elif type(self._u0[i]) == DegenerateSet:
                for j in range(self._u0[i].get_size()):
                    mag = -int(np.log10(self._u0[i].E(0)))
                    ax.plot(L, self._u0[i].E(j, L) * 10**mag, \
                            color = 'black')
                    en = self._u0[i].get_E('0', '0')
                    ax.annotate('%s' % (i), (0, en), \
                                (0, en + 5e-2 * Vmax))
                    ax.set_xlabel(r'$\lambda$')
                    ax.set_ylabel('E / $10^{%s}$' % (-mag))
            
                    
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
        self._E, self._dE2 = info[0][0], {}
        self._uD0, self._du1 = {}, {}
        for i in range(len(info)):
            self._uD0[i] = H0(IG, info[i][1])[1]
        
        self._N = len(self._uD0)
        self.W1()
        
    
    def __repr__(self):
        return 'Deg(%s, %s)' % (sigfig(self._E), self._N)
    
    
    def get_size(self):
        return len(self._uD0)
    
    
    def get_u(self, ind):
        return self._uD0new[ind]
    
    
    def get_du1(self, ind):
        return self._du1[ind]
    
    
    def get_E(self, degree, ind):
        if degree == '0':
            return self._E
        elif degree == '1':
            return self._dE1[ind]
        elif degree == '2':
            return self._dE2[ind]
        
    
    def get_R(self):
        '''
        return the resolution of the degenerate set

        Returns
        -------
        _R
            the resolution of the degenerate set. Can take 
            three possible values: '1', '2' and '0'.
                '1' - degeneracy was lifted by 1st order 
                      correction
                '2' - degeneracy was lifted by 2nd order 
                      correction
                '0' - degeneracy wasn't lifted by 1st nor
                      2nd order correction

        '''
        return self._R


    def set_du1(self, ind, du1):
        self._du1[ind] = du1
        
    
    def set_dE2(self, ind, dE2):
        self._dE2[ind] = dE2
        

    def W1(self):
        W1 = np.ones((self._N, self._N))
        for i in range(self._N):
            for j in range(self._N):
                W1[i][j] = overlap(self._uD0[i], self._IG, \
                                  self._uD0[j])
                    
        eigvec = np.linalg.eig(W1)[1]
        a, b = len(self._IG[0]), len(self._IG)
        self._dE1, self._uD0new = np.ones(len(self._uD0)), {}
         
        for i in range(len(eigvec)):
            U = np.zeros((b, a))
            for j in range(len(eigvec)):
                U += eigvec[j][i] * self._uD0[j]
            
            self._uD0new[i] = U
            self._dE1[i] = overlap(U, self._IG, U)
        
        l_bound = (0.01 / max((a, b)))**2 
        repeat = False
        for i in range(len(eigvec) - 1):
            for j in range(len(eigvec) - (i + 1)):
                if abs(self._dE1[i] - self._dE1[i + j + 1]) \
                    < l_bound:
                        repeat = True 
        
        if not repeat:
            self._R = '1'
        else:
            self._R = '2'
    
    
    def E(self, ind, l):
        return self._E + self._dE1[ind] * l + \
               self._dE2[ind] * l**2
               

    def U(self, ind, l):
        return self._uD0new[ind] + self._du1[ind] * l
    
    
    def lamb_lim(self):
        return max(abs(self._dE1/self._E))
        

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
        self._E, self._IG = info[0][0], IG
        self._u = H0(IG, info[0][1])[1]
        self._dE1 = overlap(self._u, self._IG, self._u)
        
    
    def __repr__(self):
        return 'Sin(%s)' % (sigfig(self._E))
    
    
    def get_u(self):
        return self._u
    
    
    def get_du1(self):
        return self._du1
        
    
    def get_E(self, degree):
        if degree == '0':
            return self._E
        elif degree == '1':
            return self._dE1
        elif degree == '2':
            return self._dE2
        
    
    def set_du1(self, du1):
        self._du1 = du1
        self._dE2 = overlap(self._u, self._IG, self._du1)
    
    
    def E(self, l):
        return self._E + self._dE1 * l + self._dE2 * l**2
        
    
    def U(self, l):
        return self._u + self._du1 * l
    
    
    def lamb_lim(self):
        return abs(self._dE1/self._E)
    
    
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


