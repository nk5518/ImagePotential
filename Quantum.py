# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 23:21:20 2020

@author: non_k
"""

#%%
''' Edit: 19/07/20 '''
#%%


# Schrodinger Equation


#%%
import numpy as np
import pylab as pl
from matplotlib.animation import FuncAnimation, writers 
from time import perf_counter
from functools import wraps

#%%
# Classes
@timer
class Hamiltonian:
    def __init__(self, IG, gen = 20, lim = 30):
        '''
        Parameters
        ----------
        IG : ndarray
            2D array of float, representing an image. 
        gen : int, optional
            'gen' squared eigenstates of the rectangle potential will be 
            produced. the default is 20.
        lim : int, optional
            'lim' eigenstates with higher energy and 'lim' eigenstates with 
            lower energy than the eigenstate being considered will be used to 
            calculate the correction for each eigenstate. The default is 30.

        Returns
        -------
        None.
        '''
        self._IG, self._lim = IG, lim
        S, a, b = [], len(IG[0]), len(IG)
    
        S0 = []
        for i in range(1, gen):
            for j in range(1, gen):
                S0.append((H0(IG, (i, j))[0], (i, j)))
        
        S0.sort()
        info_list, d = [], []
        min_dE = (0.01 / max((a, b)))**2
        
        for i in range(len(S0) - 1): # grouping degenerate eigenstates
            if i != len(S0) - 2:
                if S0[i + 1][0] - S0[i][0] < min_dE:
                    d.append(S0[i])
                else:
                    d.append(S0[i])
                    info_list.append(d)
                    d = []
            else:
                if S0[i + 1][0] - S0[i][0] < min_dE:
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
        '''
        Returns
        -------
        int
            the number of eigenstates with distinct energy stored within the 
            Hamiltonian instance.
        '''
        return len(self._S)
    

    def du1(self, index):
        '''
        calculate and set the coefficients for the 1st order wavefunction 
        correction for state of index 'index'.

        Parameters
        ----------
        index : int
            the index of the eigenstate whose wavefunction correction(s) are 
            to be calculated

        Returns
        -------
        None.
        '''
        S, eig, lim = self._S, self._S[index], self._lim
        if index - lim <= 0:
            s = 0
        else:
            s = index - lim
        
        if index + lim >= len(S) - 1:
            e = len(S) - 1
        else:
            e = index + lim
        R = np.arange(start = s, stop = e)
        
        set_a1 = eig.set_a1
        if type(eig) == SingleState:
            set_a1(self.out1(index, R), S)
        
        elif type(eig) == DegenerateSet:
            a1_out = {}
            for i in range(eig.size()):
                a1_out[i] = self.out1(index, R, i)
            eig.set_a1(a1_out, S)   
        eig.stat = 'complete'
        
                
    def out1(self, index, R, inner_index = 0):
        '''
        calculate the coefficients for the first order wavefunction correction,
        for wavefunction of index 'index', outside of the degenerate subspace.

        Parameters
        ----------
        index : int
            the index of the eigenstate whose wavefunction correction(s) are 
            to be calculated.
        R : ndarray
            1D array of eigenstates' indices, outside of the degenerate 
            subspace, used in the calculation.
        inner_index : int, optional
            the index of the eigenstate within the degenerate subspace. 
            the default is 0.

        Returns
        -------
        a1_out : dict
            contains the coefficients for the wavefunction correction. 
        '''
        S, IG, a1_out = self._S, self._IG, {}
        u0, En0 = S[index].get_u, S[index].get_E
        
        if type(S[index]) == SingleState:
            u_ref, e_ref = u0(0), En0(0)
        elif type(S[index]) == DegenerateSet:
            u_ref, e_ref = u0(0, inner_index), En0(0)
        
        for i in R:
            u, En = S[i].get_u, S[i].get_E
            if i != index:
                if type(S[i]) == SingleState:
                    ov = overlap(u(0), IG, u_ref)
                    de = e_ref - En(0)
                    a1_out[i] = ov / de
                    
                if type(S[i]) == DegenerateSet:
                    a1_sub = {}
                    for j in range(S[i].size()):
                        ov = overlap(u(0, j), IG, u_ref)
                        de = e_ref - En(0)
                        a1_sub[j] = ov / de
                    a1_out[i] = a1_sub                 
        return a1_out
    
    
    @timer
    def affinity(self, num = 50):
        '''
        run through the list of existing eigenstates and picks out 'num' 
        eigenstates that overlap the most with the square root of image array.

        Parameters
        ----------
        num : int, optional
            the number of eigenstates with greatest overlap to the square root
            of the image array. the default is 50.

        Returns
        -------
        list
            a list of indices of eigenstates with greatest overlap.
        '''
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
        
        A.sort()
        return [A[len(A) - 1 - i][1] for i in range(num)]
    
    
    @timer
    def Lmin(self, indices):
        indices.sort()
        S, lamb = self._S, []
        
        for i in indices:
            if S[i].stat == 'incomplete':
                self.du1(i)
            if i == 0:
                E0s = [S[i + 1].get_E(0)]
            elif i == self.size() - 1:
                E0s = [S[i - 1].get_E(0)]
            else:
                E0s = [S[i - 1].get_E(0), \
                       S[i + 1].get_E(0)]
            
            for e0 in E0s:
                Ld = S[i].L(e0)
                for l in Ld:
                    lamb.append(l)
        return min(lamb)
    
    
    # visualisation methods 
    @timer
    def display_u(self, index, saving = (False, '')):
        '''
        display a plot(s) of the perturbed eigenstate of index 'index'.

        Parameters
        ----------
        index : int
            the index of the eigenstate whose wavefunction correction(s) are
            to be calculated.
        saving : tuple, optional
            if the first element is True, the plot gets saved. the second 
            element is a string which is the name of the file to be saved.
            the default is (False, '').

        Returns
        -------
        None.
        '''
        S = self._S
        if index == self.size() - 1:
            indices = [index - 1, index]
        elif index == 0:
            indices = [index, index + 1]
        else:
            indices = [index - 1, index, index + 1]
        L = self.Lmin(indices)
        
        decor = dict(edgecolor = 'black', alpha = 0.5)
        if type(S[index]) == SingleState:
            fig, ax = pl.subplots()
            U = S[index].U(L, S)
            pl.imshow(np.conj(U) * U)
            fig.text(0.765, 0.91, '%s' % (index), bbox = decor)
            if saving[0]:
                pl.savefig(saving[1] + f'_{index}')
        
        elif type(S[index]) == DegenerateSet:
            for i in range(S[index].size()):
                fig, ax = pl.subplots()
                U = S[index].U(i, L, S)
                pl.imshow(np.conj(U) * U)
                fig.text(0.71, 0.91, '%s; %s' % (index, i), \
                         bbox = decor)
                if saving[0]:
                    pl.savefig(saving[1] + f'_{index}_{i}')
    
    
    @timer
    def energy_split(self, indices, saving = (False, '')):
        '''
        display a plot of energy splittings for eigenstates with index 
        contained within 'indices'.

        Parameters
        ----------
        indices : ndarray
            1D array containing indices of eigenstates whose energy splittings 
            are to be calculated. 
        saving : tuple, optional
            if the first element is True, the plot gets saved. the second 
            element is a string which is the name of the file to be saved.
            the default is (False, '').

        Raises
        ------
        Exception
            the index chosen is too large.
        
        Returns
        -------
        None.

        '''
        indices.sort()
        S = self._S
        if max(indices) >= len(S):
            raise Exception('please choose an index less than %s'%(len(S)))
        
        fig, ax = pl.subplots()
        V = S[indices[-1]].get_E(0)
        m = -int(np.log10(V))
        
        L = np.linspace(0, self.Lmin(indices), num = 100)
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel('E / $10^{%s}$' % (-m))
        
        for i in indices:
            en = S[i].get_E(0) * 10**m
            pos = (0, en + 0.01 * V)
            if type(S[i]) == SingleState:
                ax.annotate('%s (1)'%(i), (0, en), pos)
                ax.plot(L, S[i].E(L)*10**m, color = 'k')
                    
            elif type(S[i]) == DegenerateSet:
                size = S[i].size()
                ax.annotate('%s (%s)'%(i, size), (0, en), pos)
                for j in range(size):
                    ax.plot(L, S[i].E(j, L)*10**m, color = 'k')
        ax.autoscale()
        if saving[0]:
            pl.savefig(saving[1])
    
    
    def sup_in(self):
        '''
        sets up the initial frame for the superposition animation.

        Returns
        -------
        list
            image showing the superpostion at time t = 0.

        '''
        sU, c = self._sU, 'complex128'
        D = np.zeros((self._b, self._a), dtype = c)
        for i in sU:
            D += sU[i].astype(c)
       
        D2 = np.real(np.conj(D) * D)
        M = [max(D2[i]) for i in range(len(D2))]
        D2 *= (255 / max(M))
        self._sim.set_array(D2)
        return [self._sim] 
    
    
    def sup_up(self, t):
        '''
        updates the superpostion animation.

        Parameters
        ----------
        t : float
            time at which the evolution of the superposition is to be 
            calculated.

        Returns
        -------
        list
            image showing the superposition at time 't'.

        '''
        sU, sE, c = self._sU, self._sE, 'complex128'
        D = np.zeros((self._b, self._a), dtype = c)
        for i in sU:
            D += sU[i].astype(c) * np.exp(-1j * t * sE[i])
    
        D2 = np.real(np.conj(D) * D)
        M = [max(D2[i]) for i in range(len(D2))]
        D2 *= (255 / max(M))
        self._sim.set_array(D2)
        return [self._sim]    
            
    
    @timer
    def superpose(self, ind, weight, saving = (False, ''), repl = False):
        '''      
        display a linear superposition of the perturbed wavefunction and the 
        subsequent time evolution. 
        
        Parameters
        ----------
        ind : list/ndarray
            an array of indices of the eigenstates to be superposed.
        weight : list/ndarray
            an array of weights for each of the eigenstates to be superposed.
        saving : tuple, optional
            if the first element is True, the plot/animation will get saved.
            the second element is the name the plot/animation will get saved 
            as. the default is (False, '').
        repl : boolean, optional
            set to be True if the superpose method is being used by the 
            replicate method. the default is False.

        Raises
        ------
        Exception
            the index chosen is too large.

        Returns
        -------
        None.
        '''
        
        ind.sort()
        S, L, sqrt = self._S, self.Lmin(ind), np.sqrt 
        if max(ind) >= len(S):
            raise Exception('please choose an index less than %s'%(len(S)))
        
        if not repl:
            w1 = np.array(weight)/sqrt(np.dot(weight, weight))
            sU, sE = {}, {}
            C = 0
            for i in range(len(ind)):
                S_el = S[ind[i]]
                if type(S_el) == SingleState:
                    sU[C] = w1[i] * norm(S_el.U(L, S))
                    sE[C] = S_el.E(L) 
                    C += 1
                    
                elif type(S_el) == DegenerateSet:
                    for j in range(S_el.size()):
                        sU[C] = w1[i] * norm(S_el.U(j, L, S))
                        sE[C] = S_el.E(j, L)
                        C += 1
                
            self._sU, self._sE = sU, sE
        
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
    def replicate(self, num, saving = (False, ''), animate = True):
        '''
        decompose the image into a linear superposition of the perturbed 
        wavefunctions. 

        Parameters
        ----------
        num : int
            the number of eigenstates to be included in the superposition.
        saving : tuple, optional
            if the first element is True, the plot/animation will get saved.
            the second element is the name the plot/animation will get saved 
            as. the default is (False, '').
        animate : boolean, optional
            if True, the time evolution will be shown. the default is True.

        Returns
        -------
        None.
        '''
        ind = self.affinity(num)
        if [num, saving] != self._hist:
            self._hist = [num, saving]
            S, sqrt = self._S, np.sqrt
            L, I = self.Lmin(ind), norm(sqrt(self._IG))
    
            sU, sE = {}, {}
            C = 0
            for i in ind:
                if type(S[i]) == SingleState:
                    psi = norm(S[i].U(L, S))
                    sU[C], sE[C] = psi*overlap(psi, 1, I), S[i].E(L)
                    C += 1
                    
                elif type(S[i]) == DegenerateSet:
                    for j in range(S[i].size()):
                        psi = norm(S[i].U(j, L, S))
                        sU[C], sE[C] = psi*overlap(psi, 1, I), S[i].E(j, L)
                        C += 1 
                
            self._sU, self._sE = sU, sE
        
        if animate:
            self.superpose(ind, weight = np.ones(len(ind)), saving = saving, \
                           repl = True)
        else:
            self._sfig, self._sax = pl.subplots()
            IM0 = np.zeros((self._b, self._a))
            self._sim = self._sax.imshow(IM0, vmin = 0, \
                                         vmax = 255)
            self.sup_in()
            if saving[0]:
                pl.savefig(saving[1])
    

class SingleState:
    def __init__(self, info, IG):
        '''
        Parameters
        ----------
        info : tuple 
            the first element is the unperturbed energy eigenvalue, and the 
            second element is a tuple containing the quantum number of the 
            associated unperturbed eigenstate.
        IG : ndarray
            the image array.

        Returns
        -------
        None.
        '''
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
        self._dE2 = overlap(u(0), self._IG, norm(u(1, S)))
        
        
    @property
    def stat(self):
        return self._stat 
    
    
    @stat.setter
    def stat(self, stat):
        if stat != 'complete' and stat != 'incomplete':
            raise Exception(''' argument "stat" only accepts 'complete' or 
                            'incomplete' as input ''')
        self._stat = stat 
    
    
    def get_E(self, order):
        '''
        return energy corrections of order 'order'.

        Parameters
        ----------
        order : int
            the order of the energy correction. possible values: 0, 1 and 2. 
            0 being the unperturbed energy. 
        ind : int
            the index of the eigenstate of interest.

        Returns
        -------
        E0, dE1, or dE2 : float
            energy corrections.
        '''
        if order == 0:
            return self._E0
        elif order == 1:
            return self._dE1 
        elif order == 2:
            return self._dE2
        else:
            raise Exception('''
                            argument "order" only accepts 0, 1 or 2 as input
                            ''')
        
    
    def get_u(self, order, S = 'empty'):
        '''
        return wavefunction corrections of order 'order'. 

        Parameters
        ----------
        order : int
            the order of the wavefunction. possible values: 0 or 1. 
            0 being the unperturbed wavefunction.  
        ind : int
            the index of the eigenstate of interest.

        Returns
        -------
        u0 or du1: ndarray
            wavefunction corrections.
        '''
        IG = self._IG
        if order == 0:
            return H0(IG, self._nm)[1]
        
        elif order == 1:
            if S == 'empty':
                raise Exception('''
                                argument "states" cannot be 'empty' for 
                                "order" of 1.
                                ''')
            U1 = np.zeros((len(IG), len(IG[0])))
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
                            argument "order" only accepts 0 or 1 as input.
                            ''')
    

    def E(self, l):
        '''
        perturbative expansion for the energy of the eigenstate. 

        Parameters
        ----------
        l : TYPE
            lambda parameter which controls the strength of the pertubation.

        Returns
        -------
        ndarray
            perturbed energy for the eigenstate. calculated up to 2nd order.
        '''
        l = np.real(l)
        return self._E0 + self._dE1 * l + self._dE2 * l**2
    
    
    def U(self, l, S):
        '''
        perturbative expansion for the wavefunction of the eigenstate.

        Parameters
        ----------
        l : TYPE
            lambda parameter which controls the strength of the pertubation.

        Returns
        -------
        ndarray
            perturbed wavefunction for the eigenstate. calculated up to 1st 
            order.
        '''
        l, get_u = np.real(l), self.get_u
        return get_u(0) + get_u(1, S) * l


    def L(self, E_ext):
        ls, sqrt = [], np.sqrt
        E1, E2 = self._dE1, self._dE2
        D = (E1)**2 + 2 * E2 * (E_ext - self._E0)
        if D > 0:
            lp = (+sqrt(D) - E1) / (2 * E2)
            ln = (-sqrt(D) - E1) / (2 * E2)
            if lp > 0:
                ls.append(lp)
            if ln > 0:
                ls.append(ln)
        return ls 
    
            
class DegenerateSet:
    def __init__(self, info, IG):
        '''
        Parameters
        ----------
        info : tuple 
            the first element is the unperturbed energy eigenvalue, and the 
            second element is a tuple containing the quantum number of the 
            associated unperturbed eigenstates.
        IG : ndarray
            the image array.

        Returns
        -------
        None.
        '''
        Z = np.zeros
        self._IG, N = IG, len(info)
        self._E0, self._a1_out = info[0][0], {}
        dE1, self._dE2 = Z(N), Z(N)
        self._in, self._stat = False, 'incomplete'
        
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
            dE1[i] = overlap(norm(U), IG, norm(U))
            
        self._newb = eigvec 
        bound = (1e-2/max((len(IG), len(IG[0]))))**2
        repeat = False
        for i in range(N - 1):
            for j in range(N - (i + 1)):
                if abs(dE1[i] - dE1[i + j + 1]) < bound:
                    repeat = True 
        
        self._dE1 = dE1
        self._nm = [info[i][1] for i in range(N)]
        if repeat:
            print('E0 : %s, 2nd resolution required' % (self._E0))
            
            
    def __repr__(self):
        return 'Deg(%s, %s)' % (sigfig(self._E0), len(self._nm))
    
    
    def set_a1(self, a1_out, S):
        self._a1_out = a1_out
        size, dE1, IG = self.size(), self._dE1, self._IG
        a1_in, u = {}, self.get_u
        for i in range(size):
            a1_i_in, u_ref, e_ref = {}, u(1, i, S), dE1[i]
            self._dE2[i] = overlap(u(0, i), IG, u_ref)
            for j in range(size):
                if j != i:
                    ov = overlap(u(0, j), IG, u_ref)
                    de = e_ref - dE1[j]
                    a1_i_in[j] = ov / de 
            a1_in[i] = a1_i_in
        self._a1_in, self._in = a1_in, True 


    @property
    def stat(self):
        return self._stat
    
    
    @stat.setter
    def stat(self, stat):
        if stat != 'complete' and stat != 'incomplete':
            raise Exception('''
                            argument "stat" only accepts 'complete' or 
                            'incomplete' as input.
                            ''')
        self._stat = stat
        
    
    def size(self):
        return len(self._nm)
    
    
    def get_E(self, order, ind = 0):
        '''
        return energy corrections order 'order' of the eigenstate with index 
        'index' within the degenerate subspace.  

        Parameters
        ----------
        order : int
            the order of the correction. possible values: 0, 1 and 2. 
            0 being the unperturbed energy. 
        ind : int
            the index of the eigenstate of interest within the degenerate 
            subspace.

        Returns
        -------
        E0, dE1, or dE2 : float
            energy correction.

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
        
    
    def get_u(self, order, index, S = 'empty'):
        '''
        return wavefunction corrections order 'order' of the eigenstate with 
        index 'index' within the degenerate subspace. 

        Parameters
        ----------
        order : int
            the order of the correction. possible values: 0, 1 and 2. 
            0 being the unperturbed wavefunction.  
        index : int
            the index of the eigenstate of interest within the subspace.

        Returns
        -------
        u0, du1, or du2 : ndarray
            wavefunction correction.
        '''
        order, index = int(order), int(index)
        IG, Z = self._IG, np.zeros
        b, a = len(IG), len(IG[0])
        
        if order == 0:
            basis, U0 = self._newb, Z((b, a))
            for i in range(len(basis)):
                U0 += basis[i][index] * \
                      H0(IG, self._nm[i])[1]
            return U0
        
        elif order == 1:
            if S == 'empty':
                raise Exception('''
                                argument "states" cannot be 'empty' for 
                                "order" of 1
                                ''')
            U1 = Z((b, a))
            a1_out = self._a1_out[index]
            for i in a1_out:
                if type(S[i]) == SingleState:
                    U1 += a1_out[i] * S[i].get_u(0)
                elif type(S[i]) == DegenerateSet:
                    a1_sub = a1_out[i]
                    for j in a1_sub:
                        U1 += a1_sub[j] * S[i].get_u(0, j)
                            
            if self._in:
                a1_in = self._a1_in[index]
                for i in a1_in:
                    U1 += a1_in[i] * self.get_u(0, i)       
            return U1
        
        else:
            raise Exception('''
                            argument "order" only accepts 0 or 1 as input.
                            ''')


    def E(self, ind, l):
        '''
        pertubed energy for the degenerate eigenstate with index "ind".

        Parameters
        ----------
        ind : int
            index of the eigenstate within the degenrate subspace.
        l : float
            lambda parameter controlling the strength of perturbation.

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
        perturbed wavefunction for the degenerate eigenstate with index "ind".

        Parameters
        ----------
        ind : int
            index of a degenerate eigenstate within the degenerate subspace. 
        l : TYPE
            lambda parameter controlling the strength of the pertubation.

        Returns
        -------
        ndarray
            perturbed wavefunction for the degenerate eigenstate.
            calculated up to 1st order.
        '''
        ind = int(ind)
        return self.get_u(0, ind) + self.get_u(1, ind, S) * np.real(l)
    
    
    def L(self, E_ext):
        ls, sqrt = [], np.sqrt
        E1, E2 = self._dE1, self._dE2
        D = (E1)**2 + 2 * E2 * (E_ext - self._E0)
        for i in range(len(D)):
            if D[i] > 0:
                lp = (+sqrt(D[i]) - E1[i]) / (2 * E2[i])
                ln = (-sqrt(D[i]) - E1[i]) / (2 * E2[i])
                if lp > 0:
                    ls.append(lp)
                if ln > 0:
                    ls.append(ln)
        return ls 
  
        
# Miscellaneous Functions
def H0(F, nm): # solutions to 2D rectangle potential 
    a, b = len(F[0]), len(F)
    lin, sin, pi = np.linspace, np.sin, np.pi
    x, y = np.meshgrid(lin(0, a, num = a), \
                       lin(0, b, num = b))
    f = sin((pi / a) * nm[0] * x) * \
        sin((pi / b) * nm[1] * y)
    E = (nm[0] / a)**2 + (nm[1] / b)**2 
    return E, np.sqrt(4 / (a * b)) * f


def i_2D(Arr): # 2D integration, trapezoidal rule
    a, b = len(Arr[0]), len(Arr)
    dot, ones = np.dot, np.ones
    
    dx, dy = a / (a - 1), b / (b - 1)
    mx, my = ones(a), ones(b)
    mx[0] = mx[-1] = my[0] = my[-1] = 1/2
    mx, my = mx * dx, my * dy
    return dot(my, [dot(mx, Arr[i]) for i in range(len(Arr))])   


def overlap(bra, H, ket):  
    integrand = np.conj(bra) * H * ket 
    return i_2D(integrand)


def norm(psi):  
    return psi / np.sqrt(overlap(psi, 1, psi))
    

def sigfig(n): 
    return round(n, 3 - int(np.log10(n)))


def timer(func): 
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = perf_counter()
        value = func(*args, **kwargs)
        end = perf_counter()
        runtime = end - start
        print(f'{func.__name__} : {runtime} s')
        return value 
    return wrapper 


# processing the image 
def colour_drain(img): 
    w = [0.299, 0.5870, 0.1140]
    img_grey = np.dot(img[..., :3], w)
    return img_grey


def contract(I, N):
    '''
    reduce the size of the image array to an array of size ('N' x 'N')

    Parameters
    ----------
    I : ndarray
        image array.
    N : TYPE
        linear dimension of the new image array.

    Returns
    -------
    A : ndarray
        contracted image array.
    '''
    A = np.ones((N, N))
    rm, cm = int(np.floor(len(I) / N)), int(np.floor(len(I[0]) / N))
    
    for i in range(N):
        for j in range(N):
            a = []
            for k in range(rm):
                R0, C0, C1 = k + rm * i, cm * j, cm * (j + 1)
                a.append(sum(I[R0][C0 : C1]))
            A[i][j] = sum(a) / (rm * cm)
    return A
    

