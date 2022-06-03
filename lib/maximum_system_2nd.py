#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:14:26 2020

@author: nicolai
"""

import numpy as np
import scipy.interpolate as sp
import scipy.integrate as si

def _diff(a):
    return a[1] - a[0]

class maximum_system_2nd:
    def __init__(self, g, K, Q, T, V, I, solver='euler', vec_g=False):
        # The class encapsulates a vector-valued function defined by the equation
        #                            /                     \
        #       x(t) = sup_[r, u) Q |  K(t - s)x(s) + g(t)  |
        #                            \                     /
        # g is a vector-valued function with returns the one-sided boundary conditions
        # K is a vector-valued function which returns the single memory kernels
        # Q is a mixing matrix, which is applied after K to the resulting vector
        #   Note, that Q should already contain the sign and eigenvalue
        # T is an interval containing start and end-times
        # V is a Boolean two-vector stating if one integral limit is variable
        #   Otherwise, the limits contained in T are used
        # I is a vector of node values building the initial approximation of the solution.
        #   If it is completely zero, we build one ourselves.
        self.dim = len(K)
        self.g = g
        self.K = K
        self.Q = Q
        self.T = T
        self.V = V
        self.I = I
        self.N = len(self.I[0])
        self.ts = np.linspace(self.T[0], self.T[1], self.N)
        self.dt = _diff([self.ts[0], self.ts[1]])
        # either create the initial approximation from the boundary or copy the one provided
        self.xx = np.copy(self.I)
        # x is the interpolant while xx is the set of interpolation nodes
        # we use 'not-a-knot' boundary conditions for the splines
        self.x = [None for _ in range(self.dim)]
        for n in range(self.dim):
            self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
        self.solve = None
        if solver == 'fast_euler':
            self.solve = self.fast_backwards_euler_solve
        else:
            self.solve = solver
        # True if the given inhomogeneity is vector-valued or a vector of functions
        self.vec_g = vec_g
        # for speedup
        self.global_ts = np.arange(0., 100., self.dt)
        self.KK = np.zeros([self.dim, len(self.global_ts)])
        for n in range(self.dim):
            self.KK[n] = self.K[n](self.global_ts)

    # def banach_solve(self, tol=10**(-2), max_iter=20, normalize=False):
    #     # Iterate the points using Banach's fixed point theorem
    #     # to estimate the error, compare old and new node values
    #     err = np.inf
    #     run = 0
    #     while err > tol:
    #         old_xx = np.copy(self.xx)
    #         new_xx = [np.zeros(self.N) for _ in range(self.dim)]
    #         # update all node vectors
    #         for k in range(len(self.ts)):
    #             xx_t = self.eval_rhs(self.ts[k])
    #             for n in range(self.dim):
    #                 new_xx[n][k] = xx_t[n]
    #         self.xx = np.copy(new_xx)
    #         # update interpolants
    #         for n in range(self.dim):
    #             self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
    #         # error and max_iter updates
    #         err = np.max(np.max(np.abs(np.array(self.xx) - old_xx)))
    #         run += 1
    #         if run >= max_iter:
    #             break
    #     return (err, run)

    def fast_euler_solve(self, tol=10**(-2), max_iter=20, normalize=False):
        # A simple Euler method, which solves the maximum recursion
        max_global = np.empty(self.dim)
        # first, get the maximum described by the inhomogeneity at local time 0
        if self.vec_g:
            gg = self.g(self.ts[0])
        else:
            gg = np.array([self.g[n](self.ts[0]) for n in range(self.dim)])
        self.xx[:, 0] = gg
        for k in range(len(self.ts)):
            # first, get the maximum described by the inhomogeneity
            if self.vec_g:
                gg = self.g(self.ts[k])
            else:
                gg = np.array([self.g[n](self.ts[k]) for n in range(self.dim)])
            # now, build the local maxima und then the global maxima
            for n in range(self.dim):
                max_local = np.max(self.KK[n][k:0:-1]*self.xx[n][:k])
                max_global[n] = np.max([max_local, gg[n]])
            # now, find the maximum for state n coming from all other states through Q
            for n in range(self.dim):
                self.xx[n, k] = np.max(self.Q[n, :]*max_global)
        # update interpolants
        for n in range(self.dim):
            self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
        return (0, 0)

    # def eval_rhs(self, t, b=None):
    #     # this can be used especially for equations of the Fredholm type
    #     # to return values outside the integration limits
    #     lim = self.T
    #     if b is not None:
    #         lim = b
    #     else:
    #         if self.V[1] == True:
    #             lim[1] = t
    #         elif self.V[0] == True:
    #             lim[0] = t
    #     return self._rhs(t, lim)

    def fast_eval_rhs(self, t, b=None):
        # this can be used especially for equations of the Fredholm type
        # to return values outside the integration limits
        lim = self.T
        if b is not None:
            lim = b
        else:
            if self.V[1] == True:
                lim[1] = t
            elif self.V[0] == True:
                lim[0] = t
        return self._fast_rhs(t, lim)

    def fast_eval_arg(self, t, b=None, target=None):
        # this can be used especially for equations of the Fredholm type
        # to return values outside the integration limits
        lim = self.T
        if b is not None:
            lim = b
        else:
            if self.V[1] == True:
                lim[1] = t
            elif self.V[0] == True:
                lim[0] = t
        return self._fast_arg(t, lim, target)

    def __call__(self, t):
        # call the function x
        return np.array([self.x[n](t) for n in range(self.dim)])

    # def _rhs(self, t, lim):
    #     new_xx = np.zeros(self.dim)
    #     for n in range(self.dim):
    #         fun = (lambda s: self.K[n](t - s)*self.x[n](s))
    #         new_xx[n] = si.quad(fun, lim[0], lim[1])[0]
    #         #new_xx[n] = si.quad(fun, lim[0], lim[1], limit=3, epsabs=1e-03)[0]  # do it the rough way right now
    #     new_xx = np.dot(self.Q, new_xx)
    #     gg = np.zeros(self.dim)
    #     if self.vec_g:
    #         gg = self.g(t)
    #     else:
    #         for n in range(self.dim):
    #             gg[n] = self.g[n](t)
    #     new_xx += gg
    #     return new_xx

    def _fast_rhs(self, t, lim):
        res = np.zeros(self.dim)
        max_xx = np.zeros(self.dim)
        gg = np.zeros(self.dim)
        if self.vec_g:
            gg = self.g(t)
        else:
            for n in range(self.dim):
                gg[n] = self.g[n](t)
        for n in range(self.dim):
            u_k, l_k = self.__t2k(t, lim)
            if u_k - l_k >= self.N:
                max_xx[n] = np.max(self.KK[n][(u_k + 1):l_k:-1]*self.xx[n])
            else:
                max_xx[n] = np.max(self.KK[n][(u_k + 1):l_k:-1]*self.xx[n, :(u_k - l_k + 1)])
            max_xx[n] = np.max([max_xx[n], gg[n]])
        for n in range(self.dim):
            res[n] = np.max(self.Q[n, :]*max_xx)
        return res

    def _fast_arg(self, t, lim, target=None):
        # first, find the state, which the maximum originates from
        res = np.zeros(self.dim)
        max_xx = np.zeros(self.dim)
        # save the state idx, the origin we are getting the max from, and the time
        idx = np.zeros([self.dim, 3])
        gg = np.zeros(self.dim)
        xx_idx = np.zeros(self.dim)
        # contains the window relative to this one from the inhomgeneity and the time
        gg_info = np.zeros([self.dim, 2])
        win_idx = np.zeros(self.dim)
        if self.vec_g:
            gg, gg_info[:, 0], gg_info[:, 1] = self.g(t)
        else:
            for n in range(self.dim):
                gg[n], gg_info[n, 0], gg_info[n, 1] = self.g[n](t)
        for n in range(self.dim):
            u_k, l_k = self.__t2k(t, lim)
            if u_k - l_k >= self.N:
                arr = self.KK[n][(u_k + 1):l_k:-1]*self.xx[n]
            else:
                arr = self.KK[n][(u_k + 1):l_k:-1]*self.xx[n, :(u_k - l_k + 1)]
            xx_idx[n] = np.argmax(arr)
            max_xx[n] = arr[xx_idx[n]]
            win_idx[n] = np.argmax([max_xx[n], gg[n]])
            max_xx[n] = np.max([max_xx[n], gg[n]])
        for n in range(self.dim):
            # first, get the next state
            idx[n, 0] = np.argmax(self.Q[n, :]*max_xx)
            res[n] = np.max(self.Q[n, :]*max_xx)
            # then, find in which time interval the maximum was
            idx[n, 1] = win_idx[idx[n, 0]]
            if idx[n, 1] == 0:  # our window
                idx[n, 2] = self.ts[xx_idx[idx[n, 0]]]
            else: # one of the windows from the inhomogeneity
                idx[n, 1] = 1 + gg_info[idx[n, 0], 0]
                idx[n, 2] = gg_info[idx[n, 0], 1]
        if target is not None:
            idx = idx[target, :]
            res = res[target]
        return (res, idx)

    def __t2k(self, t, lim):
        u_k = int(np.round((t - lim[0])/self.dt, decimals=1))
        l_k = int(np.round((t - lim[1])/self.dt, decimals=1))
        return (u_k, l_k)
