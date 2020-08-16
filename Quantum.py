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
from matplotlib.animation import FuncAnimation 
import Image as Im 

#%%
# Classes
class Hamiltonian:
    def __init__(self, IG, gen = 20):
        self._IG = IG
        self._a, self._b = len(IG[0]), len(IG)
        self._S = []
        
        S0 = []
        for i in range(gen):
            for j in range(gen):
                p = (H0(IG, (i + 1, j + 1))[0], \
                     (i + 1, j + 1))
                S0.append(p)
        
        S0 = ascend(S0)
        info_list, d = [], []
        
        while len(S0) > 0:
            if len(S0) > 2:
                if abs(S0[0][0] - S0[1][0]) < \
                    (0.01 / max((self._a, self._b)))**2:
                        d.append(S0.pop(0))
                else:
                    d.append(S0.pop(0))
                    info_list.append(d)
                    d = []
                    
            elif len(S0) == 2:
                if abs(S0[0][0] - S0[1][0]) < \
                    (0.01 / max((self._a, self._b)))**2:
                        d.append(S0.pop(0))
                        d.append(S0.pop(0))
                        info_list.append(d)
                        d = []
                else:
                    d.append(S0.pop(0))
                    info_list.append(d)
                    info_list.append([S0.pop(0)])
                    d = []
        
        for i in range(len(info_list)):
            I = info_list[i]
            if len(I) == 1:
                self._S.append(SingleState(I, IG))
            elif len(I) > 1:
                self._S.append(DegenerateSet(I, IG))
    
    
    def get_size(self):
        return len(self._S)
    
    
    def du1(self, ind0, lim = 30):
        S, eig = self._S, self._S[ind0]
        if ind0 - lim <= 0:
            s = 0
        else:
            s = ind0 - lim
        
        if ind0 + lim >= len(S) - 1:
            e = len(S) - 1
        else:
            e = ind0 + lim
        R = np.arange(start = s, stop = e)
        
        if type(eig) == SingleState:
            eig.set_a1(self.out_1(eig.get_u(0), eig.get_E(0), \
                            ind0, R), S)
        
        elif type(eig) == DegenerateSet:
            for i in range(eig.get_size()):
                ui, ei = eig.get_u(0, i), eig.get_E(0, 0)
                a1_out = self.out_1(ui, ei, ind0, R) 
                eig.set_a1('out', a1_out)
                eig.set_dE2(i, overlap(ui, self._IG, \
                                       eig.get_u(1, i, S)))
                a1_in = {}
                for j in range(eig.get_size()):
                    if j != i:
                        ov = overlap(eig.get_u(0, j), self._IG, \
                                     eig.get_u(1, i, S))
                        de = eig.get_E(1, i) - eig.get_E(1, j)
                        a1_in[j] = ov / de
                eig.set_a1('in', a1_in)
        eig.set_stat()
        
                
    def out_1(self, u_ref, e_ref, ind0, R):
        '''
        calculate coefficients for the contribution of the 
        eigenstate outside of the degenerate subspace 

        Parameters
        ----------
        u_ref : ndarray
            wavefunction of the reference eigenstate 
        e_ref : float
            unperturbed energy of the reference eigenstate
        ind0 : int
            index of the reference eigenstate in _state 
        R : ndarray
            array of indices of eigenstates, outside of the 
            degenerate subspace, used in the calculation

        Returns
        -------
        a1_out : dict
            contains the coefficients for eigenstates  

        '''
        S, a1_out = self._S, {}
        for i in R:
            if i != ind0:
                if type(S[i]) == SingleState:
                    ov = overlap(S[i].get_u(0), self._IG, \
                                 u_ref)
                    de = e_ref - S[i].get_E(0)
                    a1_out[i] = ov / de
                    
                if type(S[i]) == DegenerateSet:
                    a1_sub = {}
                    for j in range(S[i].get_size()):
                        ov = overlap(S[i].get_u(0, j), \
                                     self._IG, u_ref)
                        de = e_ref - S[i].get_E(0, 0)
                        a1_sub[j] = ov / de
                    a1_out[i] = a1_sub
                    
        return a1_out
    
    
    def display_u(self, ind, lim = 30):
        S = self._S
        if S[ind].get_stat() == 'incomplete':
            self.du1(ind, lim)
        if type(S[ind]) == SingleState:
            fig, ax = pl.subplots()
            L = 0.01 * (S[ind].get_E(0) / S[ind].get_E(1))
            pl.imshow(np.conj(S[ind].U(L, S)) * S[ind].U(L, S))
            fig.text(0.71, 0.91, '%s' % (ind), \
                     bbox = dict(edgecolor = 'black', \
                                 alpha = 0.5))
        
        elif type(S[ind]) == DegenerateSet:
            L = 0.01 * (S[ind].get_E(0, 0) / S[ind].get_E(1, 0))
            for i in range(S[ind].get_size()):
                fig, ax = pl.subplots()
                pl.imshow(np.conj(S[ind].U(i, L, S)) * \
                          S[ind].U(i, L, S))
                fig.text(0.71, 0.91, '%s; %s' % (ind, i), \
                         bbox = dict(edgecolor = 'black', \
                                     alpha = 0.5))
        pl.show()
        

    def energy_split(self, indices, lim = 30):
        indices = ascend(indices)
        lmax = self.L_max(indices, lim)
        S = self._S
        
        fig, ax = pl.subplots()
        Ss = S[indices[0]]
        if type(Ss) == SingleState:
            m = -int(np.log10(Ss.get_E(0)))
            V = Ss.get_E(0)
        elif type(Ss) == DegenerateSet:
            m = -int(np.log10(Ss.get_E(0, 0)))
            V = Ss.get_E(0, 0)
        
        L = np.linspace(0, lmax, num = 100)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('E / $10^{%s}$' % (-m))
        
        for i in indices:
            if type(S[i]) == SingleState:
                en = S[i].get_E(0) * 10**m
                ax.annotate('%s (1)' % (i), (0, en), \
                            (0, en + 0.1 * V))
                ax.plot(L, S[i].E(L) * 10**m, \
                        color = 'black')
                    
            elif type(S[i]) == DegenerateSet:
                en = S[i].get_E(0, 0) * 10**m
                text = '%s (%s)' % (i, S[i].get_size())
                ax.annotate(text, (0, en), (0, en + 0.1 * V))
                for j in range(S[i].get_size()):
                    ax.plot(L, S[i].E(j, L) * 10**m, \
                            color = 'black')
        ax.autoscale()
        pl.show()
    
    
    def sup_in(self):
        sU, w1 = self._sU, self._sw1
        D = np.zeros((self._b, self._a), dtype = 'complex128')
        for i in sU:
            if type(sU[i]) == np.ndarray:
                D += sU[i] * (w1[i] + 0j)
            
            elif type(sU[i]) == dict:
                w2 = 1 / np.sqrt(len(sU[i])) 
                for j in sU[i]:
                    D += sU[i][j] * (w2 + 0j) * (w1[i] + 0j)
        
        D2 = np.real(np.conj(D) * D)
        M = 0
        for i in range(len(D2)):
            if max(D2[i]) > M:
                M = max(D2[i])
        D2 = (D2 / M) * 255
        self._sim.set_array(D2)
            
        return [self._sim] 
    
    
    def sup_up(self, t):
        sU, sE, w1 = self._sU, self._sE, self._sw1
        D = np.zeros((self._b, self._a), dtype = 'complex128')
        for i in sU:
            if type(sU[i]) == np.ndarray:
                T = np.exp(-1j * t * sE[i])
                D += sU[i] * T * (w1[i] + 0j)
            
            elif type(sU[i]) == dict:
                w2 = 1 / np.sqrt(len(sU[i]))  
                for j in sU[i]:
                    T = np.exp(-1j * t * sE[i][j])
                    D += sU[i][j] * T * (w2 + 0j) * (w1[i] + 0j)
                    
        D2 = np.real(np.conj(D) * D)
        M = 0
        for i in range(len(D2)):
            if max(D2[i]) > M:
                M = max(D2[i])
        D2 = (D2 / M) * 255
        self._sim.set_array(D2)
        
        return [self._sim] 
        
    
    def superpose(self, indices, lim = 30, weight = 'equal'):
        indices = ascend(indices)
        S, L = self._S, self.L_max(indices, lim)
        c = 'complex128'
        
        w1 = {}
        if weight == 'equal':
            for i in indices:
                w1[i] = 1 / np.sqrt(len(indices))
        elif type(weight) == list:
            D = np.sqrt(np.dot(weight, weight))
            for i in range(len(indices)):
                w1[indices[i]] = weight[i] / D
        
        sU, sE = {}, {}
        for i in indices:
            if type(S[i]) == SingleState:
                sU[i] = S[i].U(L, S).astype(c)
                sE[i] = S[i].E(L)
            elif type(S[i]) == DegenerateSet:
                sU_sub, sE_sub = {}, {}
                for j in range(S[i].get_size()):
                    sU_sub[j] = S[i].U(j, L, S).astype(c)
                    sE_sub[j] = S[i].E(j, L)
                sU[i], sE[i] = sU_sub, sE_sub
                
        self._sU, self._sE, self._sw1 = sU, sE, w1
        self._sfig, self._sax = pl.subplots()
        IM0 = np.zeros((self._b, self._a))
        self._sim = self._sax.imshow(IM0, vmin = 0, vmax = 255)
        
        if type(S[indices[0]]) == SingleState:
            Emin = S[indices[0]].get_E(0)
        elif type(S[indices[0]]) == DegenerateSet:
            Emin = S[indices[0]].get_E(0, 0)
        tau = 3 / Emin
        T = np.linspace(0, tau, num = 100)
        
        FuncAnimation(self._sfig, self.sup_up, T, self.sup_in, \
                      interval = 200, repeat = False, blit = True)


    def L_max(self, indices, lim = 30):
        indices = ascend(indices)
        E0, E1, E2, dE = [], [], [], []
        for i in indices:
            S = self._S[i] 
            if S.get_stat() == 'incomplete':
                self.du1(i, lim)
            if type(S) == SingleState:
                En = S.get_E
                E0.append(En(0))
                E1.append(abs(En(1)))
                E2.append(abs(En(2)))
            elif type(S) == DegenerateSet:
                En = S.get_E
                E0.append(En(0, 0))
                for j in range(S.get_size()):
                    E1.append(abs(En(1, j)))
                    E2.append(abs(En(2, j)))
        
        for i in range(len(E0) - 1):
            for j in range(len(E0) - (i + 1)):
                dE.append(abs(E0[i] - E0[i + j + 1]))
                
        A, B = abs(max(E1) / max(E2)), abs(min(dE) / max(E2)) 
        return 0.4 * (np.sqrt(A**2 + 4 * B) - A)                  
        
        
class SingleState:
    def __init__(self, info, IG):
        self._IG = IG
        self._E0, self._nm = info[0][0], info[0][1]
        self._dE1 = overlap(H0(IG, self._nm)[1], self._IG, \
                            H0(IG, self._nm)[1])
        self._stat = 'incomplete'
    
    def __repr__(self):
        return 'Sin(%s)' % (sigfig(self._E0))
        
        
    def set_a1(self, a1, S):
        self._a1 = a1
        self._dE2 = overlap(self.get_u(0), self._IG, \
                            self.get_u(1, S))
        
    
    def set_stat(self, stat = 'complete'):
        self._stat = stat 
        
    
    def get_stat(self):
        return self._stat 
    
    
    def get_E(self, order):
        '''
        return unperturbed energy and perturbative energy
        corrections

        Parameters
        ----------
        order : int
            the order of the energy correction. 
            Possible values: 0, 1 and 2. 
            0 being the unperturbed energy 
        ind : int
            the index of the eigenstate of interest

        Returns
        -------
        E0, dE1, or dE2 : float
            unperturbed energy or the perturbative corrections

        '''
        if order == 0:
            return self._E0
        elif order == 1:
            return self._dE1 
        elif order == 2:
            return self._dE2
        else:
            raise Exception('''
                            argument "order" only accepts 
                            0, 1 or 2 as input
                            ''')
        
    
    def get_u(self, order, S = 'empty'):
        '''
        return unperturbed eigenstate wavefunction 
        and perturbative wavefunction corrections

        Parameters
        ----------
        order : int
            the order of the wavefunction. 
            Possible values: 0, 1 and 2. 
            0 being the unperturbed wavefunction  
        ind : int
            the index of the eigenstate of interest

        Returns
        -------
        u0, du1, or du2 : ndarray
            unperturbed wavefunction or the perturbative 
            wavefunction corrections

        '''
        if order == 0:
            return H0(self._IG, self._nm)[1]
        
        elif order == 1:
            if S == 'empty':
                raise Exception('''
                                argument "states" cannot be 
                                'empty' for "order" of 1
                                ''')
            U1 = np.zeros((len(self._IG), len(self._IG[0])))
            a1 = self._a1
            for i in a1:
                if type(S[i]) == SingleState:
                    U1 += a1[i] * S[i].get_u(0)
                elif type(S[i]) == DegenerateSet:
                    a1_sub = a1[i]
                    for j in a1_sub:
                        U1 += a1_sub[j] * S[i].get_u(0, j)
            return U1
        
        else:
            raise Exception('''
                            argument "order" only accepts 
                            0 or 1 as input
                            ''')
    

    def E(self, l):
        '''
        perturbative expansion for the energy of the eigenstate 

        Parameters
        ----------
        l : TYPE
            parameter which controls the size of the pertubation

        Returns
        -------
        ndarray
            perturbed energy for the eigenstate.
            calculated up to 2nd order.

        '''
        l = np.real(l)
        return self._E0 + self._dE1 * l + self._dE2 * l**2
    
    
    def U(self, l, S):
        '''
        perturbative expansion for the wavefunction of the 
        eigenstate

        Parameters
        ----------
        l : TYPE
            parameter which controls the size of the pertubation

        Returns
        -------
        ndarray
            perturbed wavefunction for the eigenstate.
            calculated up to 1st order.

        '''
        l = np.real(l)
        return self.get_u(0) + self.get_u(1, S) * l

            
class DegenerateSet:
    def __init__(self, info, IG):
        self._IG, N = IG, len(info)
        self._E0, self._nm = info[0][0], []
        self._dE1, self._dE2 = np.zeros(N), np.zeros(N)
        self._in, self._stat = False, 'incomplete'
        for i in range(N):
            self._nm.append(info[i][1])
        
        W1 = np.ones((N, N))
        for i in range(N):
            for j in range(N):
                Ui = H0(IG, info[i][1])[1]
                Uj = H0(IG, info[j][1])[1]
                W1[i][j] = overlap(Ui, IG, Uj)
                
        eigvec = np.linalg.eig(W1)[1]
        for i in range(N):
            U = np.zeros((len(IG), len(IG[0])))
            for j in range(N):
                U += eigvec[j][i] * H0(IG, info[j][1])[1]    
            self._dE1[i] = overlap(U, IG, U)
            
        self._newb = eigvec 
        bound = (1e-2/max((len(IG), len(IG[0]))))**2
        repeat = False
        for i in range(N - 1):
            for j in range(N - (i + 1)):
                if abs(self._dE1[i] - self._dE1[i + j + 1]) < \
                   bound:
                       repeat = True 
                       
        if repeat:
            self._R = '2'
            print('2nd resolution required')
        else:
            self._R = '1'
            
    
    def __repr__(self):
        return 'Deg(%s, %s)' % (sigfig(self._E0), len(self._nm))
    
    
    def set_a1(self, in_out, A1):
        in_out_p = ['in', 'out']
        if in_out not in in_out_p:
            raise Exception('''
                            argument "in_out" only accepts 
                            'in' or 'out' as input
                            ''')
        if in_out == 'out':
            self._a1_out = A1
        elif in_out == 'in':
            self._a1_in = A1
            self._in = True
    
    
    def set_stat(self, stat = 'complete'):
        self._stat = stat
        
        
    def set_dE2(self, ind, dE2):
        '''
        set dE2 for the degenerate eigenstate with index "ind"

        Parameters
        ----------
        ind : int
            index of the degenerate eigenstate of interest
        dE2 : TYPE
            the value of 2nd order energy correction to be set 

        Returns
        -------
        None.

        '''
        self._dE2[ind] = dE2
        
    
    def get_a1(self, in_out):
        in_out_p = ['in', 'out']
        if in_out not in in_out_p:
            raise Exception('''
                            argument "in_out" only accepts 
                            'in' or 'out' as input
                            ''')
        if in_out == 'in':
            return self._a1_in
        elif in_out == 'out':
            return self._a1_out
        
        
    def get_R(self):
        '''
        return the resolution of the degenerate set 

        Returns
        -------
        _R : string
            the resolution of the degenerate set. 
            Possible values:
                '1' - 1st resolution attained
                '2' - 2nd resolution required

        '''
        return self._R
    
    
    def get_stat(self):
        return self._stat
    
    
    def get_size(self):
        return len(self._nm)
    
    
    def get_E(self, order, ind):
        '''
        return unperturbed energy and perturbative energy
        corrections

        Parameters
        ----------
        order : int
            the order of the energy correction. 
            Possible values: 0, 1 and 2. 
            0 being the unperturbed energy 
        ind : int
            the index of the eigenstate of interest

        Returns
        -------
        E0, dE1, or dE2 : float
            unperturbed energy or the perturbative corrections

        '''
        order, ind = int(order), int(ind)
        if order == 0:
            return self._E0
        elif order == 1:
            return self._dE1[ind]
        elif order == 2:
            return self._dE2[ind]
        else:
            raise Exception('''
                            argument "order" only accepts 
                            0, 1, or 2 as input
                            ''')
        
    
    def get_u(self, order, ind, S = 'empty'):
        '''
        return unperturbed eigenstate wavefunction 
        and perturbative wavefunction corrections

        Parameters
        ----------
        order : int
            the order of the wavefunction. 
            Possible values: 0, 1 and 2. 
            0 being the unperturbed wavefunction  
        ind : int
            the index of the eigenstate of interest

        Returns
        -------
        u0, du1, or du2 : ndarray
            unperturbed wavefunction or the perturbative 
            wavefunction corrections

        '''
        order, ind = int(order), int(ind)
        b, a = len(self._IG), len(self._IG[0])
        if order == 0:
            U0 = np.zeros((b, a))
            for i in range(len(self._newb)):
                U0 += self._newb[i][ind] * \
                     H0(self._IG, self._nm[i])[1]
            return U0
        
        elif order == 1:
            if S == 'empty':
                raise Exception('''
                                argument "states" cannot be 
                                'empty' for "order" of 1
                                ''')
            U1 = np.zeros((b, a))
            a1_out = self._a1_out
            for i in a1_out:
                if type(S[i]) == SingleState:
                    U1 += a1_out[i] * S[i].get_u(0)
                elif type(S[i]) == DegenerateSet:
                    a1_sub = a1_out[i]
                    for j in a1_sub:
                        U1 += a1_sub[j] * S[i].get_u(0, j)
                            
            if self._in:
                for i in self._a1_in:
                    U1 += self._a1_in[i] * self.get_u(0, i)       
            return U1
        
        else:
            raise Exception('''
                            argument "order" only accepts 
                            0 or 1 as input
                            ''')


    def E(self, ind, l):
        '''
        pertubative expansion for the energy of a degenerate 
        eigenstate with index "ind"

        Parameters
        ----------
        ind : int
            index of the degenerate eigenstate
        l : float
            parameter which controls the 'size' of perturbation

        Returns
        -------
        float
            perturbed energy for the degenerate eigenstate.
            calculated up to 2nd order. 

        '''
        ind, l = int(ind), np.real(l)
        return self._E0 + self._dE1[ind] * l + \
               self._dE2[ind] * l**2
               
    
    def U(self, ind, l, S):
        '''
        perturbative expansion for the wavefunction of a 
        degenerate eigenstate with index "ind"

        Parameters
        ----------
        ind : TYPE
            index of the degenerate eigenstate
        l : TYPE
            parameter which controls the size of the pertubation

        Returns
        -------
        ndarray
            perturbed wavefunction for the degenerate eigenstate.
            calculated up to 1st order.

        '''
        ind, l = int(ind), np.real(l)
        return self.get_u(0, ind) + self.get_u(1, ind, S) * l
  
        
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
    
    integrand = np.conj(bra) * H * ket
    
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


