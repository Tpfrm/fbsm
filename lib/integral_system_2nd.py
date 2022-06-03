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

class integral_system_2nd:
    def __init__(self, g, K, Q, T, V, I, solver='banach', vec_g=False):
        # The class encapsulates a vector-valued function defined by the equation
        #                 /u
        #       x(t) = Q  |  ds K(t - s)x(s) + g(t)
        #                 /r
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
        if not np.any(self.I):
            if vec_g:
                self.xx = np.transpose([self.g(t) for t in self.ts])
            else:
                self.xx = np.array([g_x(self.ts) for g_x in self.g])
        else:
            self.xx = np.copy(self.I)
        # x is the interpolant while xx is the set of interpolation nodes
        # we use 'not-a-knot' boundary conditions for the splines
        self.x = [None for _ in range(self.dim)]
        for n in range(self.dim):
            self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
        self.solve = None
        if solver == 'banach':
            self.solve = self.banach_solve
        elif solver == 'fast_backwards_euler':
            self.solve = self.fast_backwards_euler_solve
        else:
            self.solve = solver
        # True if the given inhomogeneity is vector-valued or a vector of functions
        self.vec_g = vec_g
        # for speedup
        self.global_ts = np.arange(0., 400., self.dt)
        self.KK = np.zeros([self.dim, len(self.global_ts)])
        for n in range(self.dim):
            self.KK[n] = self.K[n](self.global_ts)

    def banach_solve(self, tol=10**(-2), max_iter=20, normalize=False):
        # Iterate the points using Banach's fixed point theorem
        # to estimate the error, compare old and new node values
        err = np.inf
        run = 0
        while err > tol:
            old_xx = np.copy(self.xx)
            new_xx = [np.zeros(self.N) for _ in range(self.dim)]
            # update all node vectors
            for k in range(len(self.ts)):
                xx_t = self.eval_rhs(self.ts[k])
                for n in range(self.dim):
                    new_xx[n][k] = xx_t[n]
            self.xx = np.copy(new_xx)
            # update interpolants
            for n in range(self.dim):
                self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
            # error and max_iter updates
            err = np.max(np.max(np.abs(np.array(self.xx) - old_xx)))
            run += 1
            if run >= max_iter:
                break
        return (err, run)

    def fast_backwards_euler_solve(self, tol=10**(-2), max_iter=20, normalize=False):
        # A backwards Euler method obtains a solvable linear equation system for each step
        # With this method, integration for each step has to be performed only once
        # update all node vectors
        i_0 = np.empty(self.dim)
        hh = self.dt/2
        I = np.eye(self.dim)
        for k in range(len(self.ts)):
            for n in range(self.dim):
                i_0[n] = si.trapz(self.KK[n][k:0:-1]*self.xx[n][:k], dx=self.dt)
            QK = np.dot(self.Q, np.diag([self.KK[_][0] for _ in range(self.dim)]))
            if self.vec_g:
                gg = self.g(self.ts[k])
            else:
                gg = np.array([self.g[n](self.ts[k]) for n in range(self.dim)])
            self.xx[:, k] = nla.solve(I - hh*QK, gg + np.dot(self.Q, i_0 + hh*np.array([self.KK[_][1]*self.xx[_][k - 1] for _ in range(self.dim)])))
        # update interpolants
        for n in range(self.dim):
            self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
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

    def fast_eval_rhs(self, t, b=None, _banach=False):
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
        return self._fast_rhs(t, lim, _banach)

    def __call__(self, t):
        # call the function x
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
            # print('u_k: ' + str(u_k))
            # print('l_k: ' + str(l_k))
            if u_k - l_k >= self.N:
                new_xx[n] = np.trapz(self.KK[n][(u_k + 1):l_k:-1]*self.xx[n], dx=self.dt)
            else:
                new_xx[n] = np.trapz(self.KK[n][(u_k + 1):l_k:-1]*self.xx[n, :(u_k - l_k + 1)], dx=self.dt)
        # print('new_xx: ' + str(new_xx))
        # print('self.xx: ' + str(self.xx))
        new_xx = np.dot(self.Q, new_xx)
        gg = np.zeros(self.dim)
        if self.vec_g:
            gg = self.g(t)
        else:
            for n in range(self.dim):
                gg[n] = self.g[n](t)
        new_xx += gg
        return new_xx

    def __t2k(self, t):
        u_k = int(np.round((t - self.T[0])/self.dt, decimals=1))
        l_k = int(np.round((t - self.T[1])/self.dt, decimals=1))
        #print(u_k - l_k + 1)
        return (u_k, l_k)

    def __trapz_step(x, dt, n):
        int = si.trapz(x[:-1], dx=dt)
        scaling = dt/2
        x[-1] = x[-2]
        for n in range(n):
            x[-1] = int + scaling*(x[-2] + x[-1])
        return x
