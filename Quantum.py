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
from matplotlib.animation import FuncAnimation, writers 
from Image import colour_drain, contract  
from decorators import timer 
from time import perf_counter

#%%
# Classes
@timer
class Hamiltonian:
    def __init__(self, IG, gen = 20, lim = 30):
        self._IG, self._lim = IG, lim
        S, a, b = [], len(IG[0]), len(IG)
    
        S0 = []
        for i in range(1, gen):
            for j in range(1, gen):
                S0.append((H0(IG, (i, j))[0], (i, j)))

        S0 = ascend(S0)
        info_list, d = [], []
        l_bound = (0.01 / max((a, b)))**2
        
        for i in range(len(S0) - 1):
            if i != len(S0) - 2:
                if S0[i + 1][0] - S0[i][0] < l_bound:
                    d.append(S0[i])
                else:
                    d.append(S0[i])
                    info_list.append(d)
                    d = []
            else:
                if S0[i + 1][0] - S0[i][0] < l_bound:
                    d.append(S0[i])
                    d.append(S0[i + 1])
                    info_list.append(d)
                    d = []
                else:
                    d.append(S0[i])
                    info_list.append(d)
                    info_list.append([S0[i + 1]])
                    d = []
        
        for i in range(len(info_list)):
            info = info_list[i]
            if len(info) == 1:
                S.append(SingleState(info, IG))
            elif len(info) > 1:
                S.append(DegenerateSet(info, IG))
        
        self._a, self._b, self._S, self._hist = a, b, S, []
     
                
    def size(self):
        return len(self._S)
    
        
    def get_S(self):
        return self._S
    
    
    def du1(self, ind0):
        S, eig, lim = self._S, self._S[ind0], self._lim
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
            u, En = eig.get_u, eig.get_E
            for i in range(eig.size()):
                ui, ei = u(0, i), En(0, 0)
                a1_out = self.out_1(ui, ei, ind0, R) 
                eig.set_a1('out', a1_out)
                eig.set_dE2(i, overlap(ui, self._IG, \
                                       u(1, i, S)))
                a1_in = {}
                for j in range(eig.size()):
                    if j != i:
                        ov = overlap(u(0, j), self._IG, \
                                     u(1, i, S))
                        de = En(1, i) - En(1, j)
                        a1_in[j] = ov / de
                eig.set_a1('in', a1_in)
        eig.stat = 'complete'
        
                
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
                    for j in range(S[i].size()):
                        ov = overlap(S[i].get_u(0, j), \
                                     self._IG, u_ref)
                        de = e_ref - S[i].get_E(0)
                        a1_sub[j] = ov / de
                    a1_out[i] = a1_sub
                    
        return a1_out
    
    
    @timer
    def display_u(self, index, saving = (False, '')):
        S = self._S
        if index == self.size() - 1:
            indices = [index - 1, index]
        elif index == 0:
            indices = [index, index + 1]
        else:
            indices = [index - 1, index, index + 1]
        L = self.L(indices)
        
        if type(S[index]) == SingleState:
            fig, ax = pl.subplots()
            En = S[index].get_E
            L = 5e-2 * abs(En(0) / En(1))
            U = S[index].U(L, S)
            pl.imshow(np.conj(U) * U)
            fig.text(0.765, 0.91, '%s' % (index), \
                     bbox = dict(edgecolor = 'black', \
                                 alpha = 0.5))
            if saving[0]:
                pl.savefig(saving[1] + f'_{index}')
        
        elif type(S[index]) == DegenerateSet:
            En = S[index].get_E
            E0, E1 = En(0, 0), []
            for i in range(S[index].size()):
                E1.append(abs(En(1, i)))
            L = 5e-2 * (E0 / max(E1))
            
            for i in range(S[index].size()):
                fig, ax = pl.subplots()
                U = S[index].U(i, L, S)
                pl.imshow(np.conj(U) * U)
                fig.text(0.71, 0.91, '%s; %s' % (index, i), \
                         bbox = dict(edgecolor = 'black', \
                                     alpha = 0.5))
                if saving[0]:
                    pl.savefig(saving[1] + f'_{index}_{i}')
    
    
    @timer
    def energy_split(self, indices, saving = (False, '')):
        indices, S = ascend(indices), self._S
        
        fig, ax = pl.subplots()
        m = -int(np.log10(S[indices[0]].get_E(0)))
        V = S[indices[-1]].get_E(0)
        
        L = np.linspace(0, self.L(indices), num = 100)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('E / $10^{%s}$' % (-m))
        
        for i in indices:
            en = S[i].get_E(0) * 10**m
            if type(S[i]) == SingleState:
                ax.annotate('%s (1)' % (i), (0, en), \
                            (0, en + 0.1 * V))
                ax.plot(L, S[i].E(L) * 10**m, \
                        color = 'black')
                    
            elif type(S[i]) == DegenerateSet:
                text = '%s (%s)' % (i, S[i].size())
                ax.annotate(text, (0, en), (0, en + 0.1 * V))
                for j in range(S[i].size()):
                    ax.plot(L, S[i].E(j, L) * 10**m, \
                            color = 'black')
        ax.autoscale()
        if saving[0]:
            pl.savefig(saving[1])
    
    
    def sup_in(self):
        sU, w1, c = self._sU, self._sw1, 'complex128'
        D = np.zeros((self._b, self._a), dtype = c)
        for i in sU:
            D += (w1[i] + 0j) * sU[i].astype(c)
       
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
        c = 'complex128'
        D = np.zeros((self._b, self._a), dtype = c)
        for i in sU:
            exp = np.exp(-1j * t * sE[i])
            D += (w1[i] + 0j) * sU[i].astype(c) * exp
    
        D2 = np.real(np.conj(D) * D)
        M = 0
        for i in range(len(D2)):
            if max(D2[i]) > M:
                M = max(D2[i])
        D2 = (D2 / M) * 255
        self._sim.set_array(D2)
    
        return [self._sim]    
            
    
    @timer
    def superpose(self, ind, weight = 'equal', \
                  saving = (False, ''), repl = False):
        ind, sqrt = ascend(ind), np.sqrt
        S, L = self._S, self.L(ind)
        
        if not repl:
            if weight == 'equal':
                for i in ind:
                    w1 = np.ones(len(ind)) / sqrt(len(ind))
            elif type(weight) == list:
                    w1 = weight
            else:
                raise TypeError('''
                                argument "weight" only accepts 
                                'equal' or any list as input
                                ''')
    
            sU, sE = {}, {}
            C = 0
            for i in ind:
                if type(S[i]) == SingleState:
                    sU[C] = norm(S[i].U(L, S))
                    sE[C] = S[i].E(L) 
                    C += 1
                    
                elif type(S[i]) == DegenerateSet:
                    for j in range(S[i].size()):
                        sU[C] = norm(S[i].U(j, L, S))
                        sE[C] = S[i].E(j, L)
                        C += 1
                
            self._sU, self._sE, self._sw1 = sU, sE, w1
        
        self._sfig, self._sax = pl.subplots()
        IM0 = np.zeros((self._b, self._a))
        self._sim = self._sax.imshow(IM0, vmin = 0, vmax = 255)
        
        tau = 2 / min(list(self._sE.values()))
        T = np.linspace(0, tau, num = 400)
        
        A = FuncAnimation(self._sfig, self.sup_up, T, \
                          self.sup_in, interval = 100, \
                          repeat = False, blit = True)
        if saving[0]:
            write = writers['ffmpeg'](fps = 5, \
                            metadata = dict(artist = 'Me'), \
                            bitrate = 1800)
            A.save(saving[1], writer = write)


    @timer
    def replicate(self, num, saving = (False, ''), \
                  animate = True):
        sqrt = np.sqrt
        if [num, saving] != self._hist:
            self._hist = [num, saving]
            ind = self.affinity(num)
            print(ind)
            S, L, I = self._S, self.L(ind), norm(sqrt(self._IG))
                
            sU, sE, W = {}, {}, {}
            C = 0
            for i in ind:
                if type(S[i]) == SingleState:
                    psi = norm(S[i].U(L, S))
                    sU[C], sE[C] = psi, S[i].E(L)
                    W[C] = overlap(psi, 1, I)
                    C += 1
                    
                elif type(S[i]) == DegenerateSet:
                    for j in range(S[i].size()):
                        psi = norm(S[i].U(j, L, S))
                        sU[C], sE[C] = psi, S[i].E(j, L)
                        W[C] = overlap(psi, 1, I)
                        C += 1 
                
            self._sU, self._sE, self._sw1 = sU, sE, W
        
        if animate:
            self.superpose(ind, saving = saving, \
                           repl = True)
        else:
            self._sfig, self._sax = pl.subplots()
            IM0 = np.zeros((self._b, self._a))
            self._sim = self._sax.imshow(IM0, vmin = 0, \
                                         vmax = 255)
            self.sup_in()
            if saving[0]:
                pl.savefig(saving[1])
    
    
    @timer
    def affinity(self, num = 50):
        sqrt = np.sqrt
        S, I, A = self._S, norm(sqrt(self._IG)), []
        
        for i in range(len(S)):
            if type(S[i]) == SingleState:
                U = S[i].get_u(0)
                c = abs(overlap(U, 1, I))
                A.append((c, i))
        
            elif type(S[i]) == DegenerateSet:
                c = 0
                for j in range(S[i].size()):
                    U = S[i].get_u(0, j)
                    c += abs(overlap(U, 1, I))
                A.append((c, i))
        
        A = ascend(A)
        end, ind = len(A) - 1, []
        for i in range(num):
            ind.append(A[end - i][1])
        
        return ind 
    
    
    @timer
    def L(self, indices):
        S = self._S
        indices, lamb, sqrt = ascend(indices), [], np.sqrt
        for i in indices:
            if S[i].stat == 'incomplete':
                self.du1(i)
            E0 = S[i].get_E(0)
            if i == 0:
                E0s = [S[i + 1].get_E(0)]
            elif i == self.size() - 1:
                E0s = [S[i - 1].get_E(0)]
            else:
                E0s = [S[i - 1].get_E(0), \
                       S[i + 1].get_E(0)]
            
            for e0 in E0s:
                if type(S[i]) == SingleState:
                    En = S[i].get_E
                    E1, E2 = En(1), En(2)
                    D = (E1)**2 + 2 * E2 * (e0 - E0)
                    if D > 0:
                        lp = (sqrt(D) - E1) / (2 * E2)
                        ln = (-sqrt(D) - E1) / (2 * E2)
                        if lp > 0:
                            lamb.append(lp)
                        if ln > 0:
                            lamb.append(ln)
                
                elif type(S[i]) == DegenerateSet:
                    for j in range(S[i].size()):
                        En = S[i].get_E
                        E1, E2 = En(1, j), En(2, j)
                        D = (E1)**2 + 2 * E2 * (e0 - E0)
                        if D > 0:
                            lp = (sqrt(D) - E1) / (2 * E2)
                            ln = (-sqrt(D) - E1) / (2 * E2)
                            if lp > 0:
                                lamb.append(lp)
                            if ln > 0:
                                lamb.append(ln)            
        return min(lamb)
    

class SingleState:
    def __init__(self, info, IG):
        self._IG = IG
        self._E0, self._nm = info[0]
        u = H0(IG, info[0][1])[1]
        self._dE1 = overlap(u, IG, u)
        self._stat = 'incomplete'
    
    def __repr__(self):
        return 'Sin(%s)' % (sigfig(self._E0))
        
        
    def set_a1(self, a1, S):
        self._a1 = a1
        u = self.get_u
        self._dE2 = overlap(u(0), self._IG, u(1, S))
        
        
    @property
    def stat(self):
        return self._stat 
    
    
    @stat.setter
    def stat(self, stat):
        if stat != 'complete' and stat != 'incomplete':
            raise Exception('''
                            argument "stat" only accepts 
                            'complete' or 'incomplete' as 
                            input
                            ''')
        self._stat = stat 
    
    
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
        l, get_u = np.real(l), self.get_u
        return get_u(0) + get_u(1, S) * l

            
class DegenerateSet:
    def __init__(self, info, IG):
        Z = np.zeros
        self._IG, N = IG, len(info)
        self._E0, self._nm = info[0][0], []
        self._dE1, self._dE2 = Z(N), Z(N)
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
            U = Z((len(IG), len(IG[0])))
            for j in range(N):
                U += eigvec[j][i] * H0(IG, info[j][1])[1]    
            self._dE1[i] = overlap(norm(U), IG, norm(U))
            
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
            print('E0 : %s, 2nd resolution required' \
                  % (self._E0))
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
    
    
    @property
    def stat(self):
        return self._stat
    
    
    @stat.setter
    def stat(self, stat):
        if stat != 'complete' and stat != 'incomplete':
            raise Exception('''
                            argument "stat" only accepts 
                            'complete' or 'incomplete' as 
                            input
                            ''')
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

    
    def size(self):
        return len(self._nm)
    
    
    def get_E(self, order, ind = 0):
        '''
        return unperturbed energy or perturbed energy 
        corrections for degenerative set 

        Parameters
        ----------
        order : int
            the order of correction. 
            Possible values: 0, 1 and 2. 
            0 being the unperturbed energy 
        ind : int
            the index of the eigenstate of interest

        Returns
        -------
        E0, dE1, or dE2 : float
            unperturbed energy or the perturbed energy 
            corrections

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
        return unperturbed wavefunction or perturbed  
        corrections for degenerative set

        Parameters
        ----------
        order : int
            the order of the correction. 
            Possible values: 0, 1 and 2. 
            0 being the unperturbed wavefunction  
        ind : int
            the index of the eigenstate of interest

        Returns
        -------
        u0, du1, or du2 : ndarray
            unperturbed wavefunction or perturbed 
            corrections

        '''
        order, ind = int(order), int(ind)
        IG, Z = self._IG, np.zeros
        b, a = len(IG), len(IG[0])
        if order == 0:
            basis = self._newb
            U0 = Z((b, a))
            for i in range(len(basis)):
                U0 += basis[i][ind] * \
                     H0(IG, self._nm[i])[1]
            return U0
        
        elif order == 1:
            if S == 'empty':
                raise Exception('''
                                argument "states" cannot be 
                                'empty' for "order" of 1
                                ''')
            U1 = Z((b, a))
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
        pertubative energy for the degenerate eigenstate 
        with index "ind"

        Parameters
        ----------
        ind : int
            index of the degenerate eigenstate
        l : float
            parameter controlling the 'size' of perturbation

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
        perturbed wavefunction for the degenerate eigenstate 
        with index "ind"

        Parameters
        ----------
        ind : TYPE
            index of the degenerate eigenstate
        l : TYPE
            parameter controlling the 'size' of the pertubation

        Returns
        -------
        ndarray
            perturbed wavefunction for the degenerate eigenstate.
            calculated up to 1st order.

        '''
        ind, l = int(ind), np.real(l)
        get_u = self.get_u
        return get_u(0, ind) + get_u(1, ind, S) * l
  
        
# Miscellaneous Functions
def H0(F, nm):
    a, b = len(F[0]), len(F)
    lin, sin, pi = np.linspace, np.sin, np.pi
    x, y = np.meshgrid(lin(0, a, num = a), \
                       lin(0, b, num = b))
    f = sin((pi / a) * nm[0] * x) * \
        sin((pi / b) * nm[1] * y)
    E = (nm[0] / a)**2 + (nm[1] / b)**2 
        
    return E, np.sqrt(4 / (a * b)) * f


def i_2D(Arr):
    a, b = len(Arr[0]), len(Arr)
    dot, ones = np.dot, np.ones
    
    dx, dy = a / (a - 1), b / (b - 1)
    mx, my = ones(a), ones(b)
    mx[0] = mx[-1] = my[0] = my[-1] = 1/2
    mx, my = mx * dx, my * dy
    
    V = []
    for i in range(len(Arr)):
        V.append(dot(mx, Arr[i]))

    return dot(my, V)   


def overlap(bra, H, ket):
    if type(H) == int:
        H = np.ones((len(bra), len(bra[0])))
    
    integrand = np.conj(bra) * H * ket
    
    return i_2D(integrand)


def norm(psi):
    return psi / np.sqrt(overlap(psi, 1, psi))
    
    
def ascend(o):
    o_type = [float, int, tuple]
    if type(o[0]) not in o_type:
        raise TypeError('''
                        elements of "o" must either be all 
                        floats, all ints, or all tuples 
                        ''')
    N = len(o)
    for i in range(N - 1):
        for j in range(N - i - 1):
            if type(o[0]) == tuple:
                if o[j][0] > o[j + 1][0]:
                    o[j], o[j + 1] = o[j + 1], o[j]
            else:
                if o[j] > o[j + 1]:
                    o[j], o[j + 1] = o[j + 1], o[j]
    
    return o


def sigfig(n):   
    return round(n, 3 - int(np.log10(n)))


#%%
img = im.imread('alex.jpg')
img_g = colour_drain(img)
I = contract(img_g)

h = Hamiltonian(I, gen = 30)
print(h.size())

#%%
#h.energy_split([50, 51, 52])
#h.display_u(100)
#h.superpose([0, 50, 100, 150, 200, 250, 300, 350], \
#           saving = (True, 'lenna.mp4'))
h.replicate(num = 100, animate = False)

#%%
I_sqrt = norm(np.sqrt(I))
S, E0, A = h.get_S(), [], []
start = perf_counter()
for i in range(len(S)):
    if type(S[i]) == SingleState:
        U = S[i].get_u(0)
        c = abs(overlap(U, 1, I_sqrt))
        A.append((c, i))
        E0.append(S[i].get_E(0))
        
    elif type(S[i]) == DegenerateSet:
        c = 0
        for j in range(S[i].size()):
            U = S[i].get_u(0, j)
            c += abs(overlap(U, 1, I_sqrt))
        A.append((c, i))
        E0.append(S[i].get_E(0))
end = perf_counter()
print(f'affinity : {end - start} s')
A = ascend(A)

ind, end = [], len(A) - 1
for i in range(250):
    ind.append(A[end - i][1])

L, c, Z = h.get_L(), 'complex128', np.zeros
U, E = {}, {}
count = 0
start = perf_counter()
for i in ind:
    if S[i].stat == 'incomplete':
        h.du1(i)
    if type(S[i]) == SingleState:
        U[count] = norm(S[i].U(L, S))
        E[count] = S[i].E(L)
        count += 1 
    elif type(S[i]) == DegenerateSet:
        for j in range(S[i].size()):
            U[count] = norm(S[i].U(j, L, S))
            E[count] = S[i].E(j, L)
            count += 1 
end = perf_counter()
print(f'wavefunction and energy : {end - start} s')
#pl.plot([i for i in range(len(S))], E0)
#%%
#static
wvfunc = Z((255, 255))
for i in U: 
    wvfunc += overlap(U[i], 1, I_sqrt) * U[i]
    
pl.imshow(wvfunc * wvfunc)

#%%
#animate 
# IT WORKKKKKKSSSSSSS!!!!!!!!!!!
# The animation should be like this
fig, ax = pl.subplots()
film = ax.imshow(np.zeros((255, 255)), vmin = 0, vmax = 255)

W = {}
for i in U:
    W[i] = overlap(U[i], 1, I_sqrt)

def init():
    D = np.zeros((255, 255))
    for i in U:
        D += W[i] * U[i]
    
    D2 = np.real(np.conj(D) * D)
    M = 0
    for i in range(len(D2)):
        if max(D2[i]) > M:
            M = max(D2[i])
    D2 = (D2 / M) * 255
    film.set_array(D2)
    
    return [film]


def update(t):
    D = np.zeros((255, 255), dtype = c)
    for i in U:
        exp = np.exp(-1j * t * E[i])
        D += (W[i] + 0j) * U[i].astype(c) * exp
    
    D2 = np.real(np.conj(D) * D)
    M = 0
    for i in range(len(D2)):
        if max(D2[i]) > M:
            M = max(D2[i])
    D2 = (D2 / M) * 255
    film.set_array(D2)
    
    return [film]    

tau = 2 / min(list(E.values()))
T = np.linspace(0, tau, num = 500)
A = FuncAnimation(fig, update, T, init, interval = 100, \
              repeat = False, blit = True)
write = writers['ffmpeg'](fps = 5, \
                          metadata = dict(artist = 'Me'), \
                          bitrate = 1800)
A.save('alex.mp4', writer = write)

#%%
S = h.get_S()
fig = pl.figure()
for i in range(len(S)):
    pl.hlines(S[i].get_E(0), xmin = i - 10, xmax = i + 10)

#%%
S = h.get_S()
ind, E1, E2 = [], [], []
for i in range(len(S)):
    if S[i].stat == 'incomplete':
        h.du1(i)
    if type(S[i]) == SingleState:
        En = S[i].get_E
        E1.append(En(1))
        E2.append(En(2))
        ind.append(i)
    elif type(S[i]) == DegenerateSet:
        En = S[i].get_E
        for j in range(S[i].size()):
            E1.append(En(1, j))
            E2.append(En(2, j))
            ind.append(i)

pl.figure()
pl.title('E1')
pl.plot(ind, E1, label = '627')
pl.legend()
pl.savefig('E1')

pl.figure()
pl.title('E2')
pl.plot(ind, E2, label = '627')
pl.legend()
pl.savefig('E2')

#%%
S, sqrt = h.get_S(), np.sqrt
E0, e0 = S[139].get_E(0), S[138].get_E(0)
dE = e0 - E0
for i in range(S[139].size()):
    E1, E2 = S[139].get_E(1, i), S[139].get_E(2, i) 
    D = E1**2 + 2 * E2 * dE
    print((sqrt(D) - E1) / (2 * E2))
    print((-sqrt(D) - E1) / (2 * E2))


