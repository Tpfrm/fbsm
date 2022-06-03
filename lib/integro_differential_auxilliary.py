#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:14:26 2020

@author: nicolai
"""

import numpy as np
from copy import deepcopy
import scipy.interpolate as sp
import scipy.integrate as si
from lib.integral_system_2nd import integral_system_2nd
from lib.integrodifferential_system_2nd import integrodifferential_system_2nd
from lib.integrodifferential_ctsmc import integrodifferential_CTSMC
from lib.integral_ctsmc import integral_CTSMC

# integrodifferential_to_integral_system (intdif, times_valid, acc)
#
#   intdif: integrodifferential_system_2nd object
#   times_valid: in which time interval the result shall be queried
#   acc: how many points shall be used for spline integration within times_valid
#
# This function takes an integrodifferential system and returns the respective
# integral system. However, it will not be checked, if there arise any issues with
# convergence
# T gives the time horizon. Out of this horizon, the new system cannot be used.
def integrodifferential_to_integral_system(intdif, times_valid, acc):
    dim = intdif.dim
    ts = np.linspace(times_valid[0], times_valid[1], acc)
    g = [None for _ in range(dim)]
    K = [None for _ in range(dim)]
    for n in range(dim):
        gs = np.zeros(acc)
        ks = np.zeros(acc)
        # sample the functions g and K
        for k in range(acc):
            gs[k] = intdif.g(ts[k])
            ks[k] = intdif.K(ts[k])
        # integrate g
        g[n] = sp.CubicSplines(ts, gs).antiderivative()
        # resample the integral of g and add the offset I on it
        for k in range(acc):
            gs[k] = g[n](ts[k]) + intdif.I[n]
        # finalize both integrals of K and g
        g[n] = sp.CubicSplines(ts, gs)
        K[n] = sp.CubicSplines(ts, ks).antiderivative()
    Q = intdif.Q
    T = intdif.T
    V = intdif.V
    I = np.zeros(dim, intdif.N)
    return integral_system_2nd(g, K, Q, T, V, I)

#_______________________________________________________________________________


# memory_kernels_from_sjts (sjt_dists)
def memory_kernels_from_sjts(sjt_dists):
    dim = len(sjt_dists)
    ts = np.linspace(0, 10, 100)

    # get sojourn time distributions
    sjt_f = [None for _ in range(dim)]
    sjt_df = [None for _ in range(dim)]
    for s in range(dim):
        sjt_f[s] = sjt_dists[s].fun
        sjt_df[s] = sp.CubicSpline(ts, sjt_f[s](ts)).derivative(1)

    # build the integrodifferential system defining the kernels
    g = [None for _ in range(dim)]
    for n in range(dim):
        g[n] = (lambda t, n=n: sjt_df[n](t))
    kernels = integral_system_2nd(g=g, K=sjt_f, Q=-np.eye(dim), T=[0, 10], V=[0, 1], I=np.zeros([dim, 50]))
    (err, run) = kernels.solve(normalize=False)
    print(err)
    print(run)

    # return the interpolated kernels
    ks = [None for _ in range(dim)]
    for n in range(dim):
        ks[n] = kernels.x[n]
    return ks

#_______________________________________________________________________________


# discretizedCTSMC_to_integrodifferentialCTSMC (hsmm)
def discretizedCTSMC_to_integrodifferentialCTSMC(dctsmc):
    # get initial state
    p_0 = np.copy(dctsmc.p_0)
    ictsmc = integrodifferential_CTSMC(p_0)
    # get sojourn time distributions and embedded Markov chain
    dim = dctsmc.num_states
    for s in range(dim):
        ictsmc.set_sjt_dist(s, deepcopy(dctsmc.sjt_dists[s]))
        ictsmc.set_jp_dist(s, deepcopy(dctsmc.jp_dists[s]))
    ictsmc.finalize()
    return ictsmc

# discretizedCTSMC_to_integrodifferentialCTSMC (hsmm)
def discretizedCTSMC_to_integralCTSMC(dctsmc):
    # get initial state
    p_0 = np.copy(dctsmc.p_0)
    ictsmc = integral_CTSMC(p_0)
    # get sojourn time distributions and embedded Markov chain
    dim = dctsmc.num_states
    for s in range(dim):
        ictsmc.set_sjt_dist(s, deepcopy(dctsmc.sjt_dists[s]))
        ictsmc.set_jp_dist(s, deepcopy(dctsmc.jp_dists[s]))
    ictsmc.finalize()
    return ictsmc

# integrodifferentialCTSMC_to_HSMM (intdif)
def integrodifferentialCTSMC_to_HSMM(intdif):
    pass
