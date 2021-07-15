# -*- coding: utf-8 -*-
"""
Created on Thu Jul 15 01:17:48 2021

@author: non_k
"""

#%%
import Quantum as Q
import matplotlib.image as im 

N = 255
img = Q.colour_drain(im.imread('lenna.jpg'))
if min((len(img), len(img[0]))) > N:
    img = Q.contract(img, N)

h = Q.Hamiltonian(img, gen = 30)
print('%s unperturbed eigenstates with distinct energies were generated \
      initially' % (h.size()))

#%%
''' 
    display the energies of chosen eigenstates as a function of the lambda 
    parameter.
'''

h.energy_split([150, 151, 152]) 

#%%
'''
    display a perturbed wavefunction squared associated with the 
'''

h.display_u(177) 

#%%
'''
    display a linear superposition of perturbed wavefunctions and the 
    subsequent time evolution. 
'''

h.superpose([0, 50, 100, 150, 200], np.ones(5)) 

#%%
'''
    decompose the image on the perturbed wavefunctions. 
    
    assigning animate = True will result in the time evolution being shown as 
    well, but the execution time will be much longer.  
    
    recommend saving the animation so you will not miss it once the execution
    finished 
'''

h.replicate(num = 150, animate = False)
#h.replicate(num = 150, animate = True, saving = (True, 'lenna.mp4'))


