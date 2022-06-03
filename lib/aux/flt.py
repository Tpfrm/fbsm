#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:14:26 2020

@author: nicolai
"""

import numpy as np
import scipy.fft as sfft
from scipy.fft import rfft, irfft

# this file contains functions executing a afap (one-sided) laplace transform
# of a real signal employing the speed of the fft implementation of the dct

sigma = 0.

def flt(x, ts):
    xx = np.append(np.zeros(1), x*np.exp(-sigma*ts))
    N = sfft.next_fast_len(len(xx), True)
    y = rfft(xx, n=N)
    return y

def iflt(y, ts):
    x = irfft(y)
    x = x[1:(len(ts) + 1)]*np.exp(sigma*ts)
    return x
