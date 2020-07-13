# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 00:26:23 2020

@author: non_k
"""

#%%
import numpy as np
import matplotlib.image as im 
import pylab as pl 
import numpy.fft as fo

#%%
def colour_drain(img):
    w = [0.299, 0.5870, 0.1140]
    img_grey = np.dot(img[..., :3], w)
    
    return img_grey


def circle_filter(img_grey, ftype, cut_off):
    if ftype != 'high' and ftype != 'low':
        raise Exception("ftype can only be 'high' or 'low'")
    
    row, col = img_grey.shape    
    F_shift = fo.fftshift(fo.fft2(img_grey))
    
    for i in range(len(F_shift)):
        for j in range(len(F_shift[0])):
            if ftype == 'low':
                if (i - row/2)**2 + (j - col/2)**2 > \
                    (row/cut_off)**2:
                    F_shift[i][j] = 0 
            elif ftype == 'high':
                if (i - row/2)**2 + (j - col/2)**2 < \
                    (row/cut_off)**2:
                    F_shift[i][j] = 0 
    
    F_ishift = fo.ifftshift(F_shift)
    img_back = abs(fo.ifft2(F_ishift))
    
    return img_back


