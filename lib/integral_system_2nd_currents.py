#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:14:26 2020

@author: nicolai
"""

import numpy as np
import scipy.interpolate as sp
import scipy.integrate as si
import numpy.linalg as nla
import matplotlib.pyplot as plt

def _diff(a):
    return a[1] - a[0]

class integral_system_2nd_currents:
    def __init__(self, g, K, Q, T, N, solver='fast_backwards_euler', vec_g=False):
        # The class encapsulates a vector-valued function defined by the equation
        #                 /  /u                       \
        #       x(t) = Q |   |  ds K(t - s)x(s) + g(t) |
        #                 \  /r                       /
        # g is a vector-valued function with returns the one-sided boundary conditions
        # K is a vector-valued function which returns the single memory kernels
        # Q is a mixing matrix, which is applied after K to the resulting vector
        #   Note, that Q should already contain the sign and eigenvalue
        # T is an interval containing start and end-times
        # V is a Boolean two-vector stating if one integral limit is variable
        #   Otherwise, the limits contained in T are used
        # I is a vector of node values building the initial approximation of the solution.
        #   If it is completely zero, we build one ourselves.
        # In this class, solving the original equation also solves the embedded
        # function in the brackets. In a CTSMC context, this allows for a simultaneous
        # solution of input and output currents
        # note, that it doesn't support initial approximations and I is thus only
        # included for compatibility reasons
        self.dim = len(K)
        self.g = g
        self.K = K
        self.Q = Q
        self.T = T
        self.N = N
        self.ts = np.linspace(self.T[0], self.T[1], self.N)
        self.dt = _diff([self.ts[0], self.ts[1]])
        # create the initial approximation from the boundary
        self.xx = np.zeros([self.dim, len(self.ts)])
        if vec_g:
            gg = self.g(self.ts[0])
        else:
            gg = np.array([self.g[n](self.ts[0]) for n in range(self.dim)])
        self.xx[:, 0] = np.copy(np.dot(self.Q, gg))
        # x is the interpolant while xx is the set of interpolation nodes
        # we use 'not-a-knot' boundary conditions for the splines
        self.x = [None for _ in range(self.dim)]
        self.solve = None
        # if solver == 'banach':
        #     self.solve = self.banach_solve
        if solver == 'fast_backwards_euler':
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
        # Also allocate nodes and an interpolant object for the output currents z
        self.zz = np.zeros([self.dim, len(self.ts)])
        self.z = [None for _ in range(self.dim)]
        self.zz[:, 0] = np.copy(gg)
        # allocate an interpolant object for the integrated marginal
        self.p = [None for _ in range(self.dim)]

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

    def fast_backwards_euler_solve(self, tol=10**(-2), max_iter=20, normalize=False):
        # A backwards Euler method obtains a solvable linear equation system for each step
        # With this method, integration for each step has to be performed only once
        # update all node vectors
        i_0 = np.empty(self.dim)
        hh = self.dt/2
        I = np.eye(self.dim)
        for k in range(len(self.ts)):
            t = self.ts[k]
            for n in range(self.dim):
                i_0[n] = si.trapz(self.KK[n][k:0:-1]*self.xx[n][:k], dx=self.dt)
            QK = np.dot(self.Q, np.diag([self.KK[_][0] for _ in range(self.dim)]))
            if self.vec_g:
                gg = self.g(t)
            else:
                gg = np.array([self.g[n](t) for n in range(self.dim)])
            rhs_woQ = gg + i_0 + hh*np.array([self.KK[_][1]*self.xx[_][k - 1] for _ in range(self.dim)])
            self.xx[:, k] = nla.solve(I - hh*QK, np.dot(self.Q, rhs_woQ))
            self.zz[:, k] = rhs_woQ + hh*np.array([self.KK[_][0]*self.xx[_][k] for _ in range(self.dim)])
        #print(self.xx[:, 0])
        # update interpolants
        #plt.figure()
        for n in range(self.dim):
            self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
            #plt.plot(self.ts, self.x[n](self.ts))
            #print(self.x[n](self.ts[-1]))
            self.z[n] = sp.CubicSpline(self.ts, self.zz[n])
            self.p[n] = sp.CubicSpline(self.ts, self.xx[n] - self.zz[n]).antiderivative()
        #plt.show()
        return (0, 0)

    def eval_rhs(self, t, b=None):
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
        return self._rhs(t, lim)

    def fast_eval_rhs(self, k, b=None):
        # this can be used especially for equations of the Fredholm type
        # to return values outside the integration limits
        lim = [0, len(self.ts) - 1]
        if b is not None:
            lim = b
        else:
            if self.V[1] == True:
                lim[1] = k
            elif self.V[0] == True:
                lim[0] = k
        return self._fast_rhs(k, lim)

    def __call__(self, t, z=False):
        # z can be requested
        if z:
            return np.array([self.z[n](t) for n in range(self.dim)])
        # return np.array([self.p[n](t) for n in range(self.dim)])
        return np.array([self.x[n](t) for n in range(self.dim)])

    def _rhs(self, t, lim):
        new_xx = np.zeros(self.dim)
        for n in range(self.dim):
            fun = (lambda s: self.K[n](t - s)*self.x[n](s))
            new_xx[n] = si.quad(fun, lim[0], lim[1])[0]
            #new_xx[n] = si.quad(fun, lim[0], lim[1], limit=3, epsabs=1e-03)[0]  # do it the rough way right now
        new_xx = np.dot(self.Q, new_xx)
        gg = np.zeros(self.dim)
        if self.vec_g:
            gg = self.g(t)
        else:
            for n in range(self.dim):
                gg[n] = self.g[n](t)
        new_xx += gg
        return new_xx

    def _fast_rhs(self, t, lim, _banach=False):
        new_xx = np.zeros(self.dim)
        for n in range(self.dim):
            #fun = (lambda s: self.K[n](t - s)*self.x[n](s))
            #new_xx[n] = si.quad(fun, lim[0], lim[1])[0]
            #new_xx[n] = si.quad(fun, lim[0], lim[1], limit=3, epsabs=1e-03)[0]  # do it the rough way right now
            u_k, l_k = self.__t2k(t)
            new_xx[n] = si.trapz(self.KK[n][(u_k + 1):l_k:-1]*self.xx[n], dx=self.dt)
        gg = np.zeros(self.dim)
        if self.vec_g:
            gg = self.g(t)
        else:
            for n in range(self.dim):
                gg[n] = self.g[n](t)
        new_xx += gg
        new_xx = np.dot(self.Q, new_xx)
        return new_xx

    def __t2k(self, t):
        u_k = int(np.round((t - self.T[0])/self.dt, decimals=5))
        l_k = int(np.round((t - self.T[1])/self.dt, decimals=5))
        return (u_k, l_k)

    def __trapz_step(x, dt, n):
        int = si.trapz(x[:-1], dx=dt)
        scaling = dt/2
        x[-1] = x[-2]
        for n in range(n):
            x[-1] = int + scaling*(x[-2] + x[-1])
        return x
