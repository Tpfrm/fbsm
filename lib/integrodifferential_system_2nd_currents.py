#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 11:14:26 2020

@author: nicolai
"""

import numpy as np
import scipy.interpolate as sp
import scipy.integrate as si
from idesolver import IDESolver

def _diff(a):
    return a[1] - a[0]

class integrodifferential_system_2nd_currents:
    def __init__(self, g, K, Q, T, V, I, N, solver='fast_euler', vec_g=False):
        # The class encapsulates a vector-valued function defined by the equation
        #                 /  /u                       \
        #   d_t x(t) = Q |   |  ds K(t - s)x(s) + g(t) |
        #                 \  /r                       /
        # g is a vector-valued function with returns the one-sided boundary conditions
        # K is a vector-valued function which returns the single memory kernels
        # Q is a mixing matrix, which is applied after K to the resulting vector
        #   Note, that Q should already contain the sign and eigenvalue
        # T is an interval containing start and end-times
        # V is a Boolean two-vector stating if one integral limit is variable
        #   Otherwise, the limits contained in T are used
        # I is a vector of initial values, where the integration starts
        # solver is either a string or a function which acts as the solver
        self.dim = len(K)
        self.g = g
        self.K = K
        self.Q = Q
        self.T = T
        self.V = V
        self.N = N
        self.ts = np.linspace(self.T[0], self.T[1], self.N)
        self.local_ts = np.linspace(0, self.T[1] - self.T[0], self.N)
        self.dt = _diff([self.ts[0], self.ts[1]])
        self.I = I
        # x is the interpolant while xx is the set of interpolation nodes
        # z is the interpolant while zz is the set of interpolation nodes
        # we use 'not-a-knot' boundary conditions for the splines
        self.xx = [np.zeros(N) for _ in range(self.dim)]
        self.zz = [np.zeros(N) for _ in range(self.dim)]
        self.x = [None for _ in range(self.dim)]
        self.z = [None for _ in range(self.dim)]
        self.solve = None
        # if solver == 'karpel':
        #     self.solve = self.karpel_solve
        # elif solver == 'euler':
        #     self.solve = self.euler_solve
        if solver == 'fast_euler':
            self.solve = self.fast_euler_solve
        else:
            self.solve = solver
        # True if the given inhomogeneity is vector-valued or a vector of functions
        self.vec_g = vec_g
        # for speedup
        self.KK = np.zeros([self.dim, len(self.local_ts)])
        for n in range(self.dim):
            self.KK[n] = self.K[n](self.local_ts)

    # def euler_solve(self, tol=10**(-2), max_iter=20, normalize=False):
    #     # Use a simple multistep backward-Euler scheme to integrate the target function
    #     # as an integrator, simply use Simpson's rule on the already obtained steps
    #     # error estimates are not yet available
    #
    #     #print(self.g(self.ts[0]))
    #     self.xx = np.array([np.concatenate((np.array([self.I[n], self.I[n]]), np.zeros(self.N - 2))) for n in range(self.dim)])
    #     #self.xx[:, 0] += self.g(self.ts[0])
    #     if self.vec_g:
    #         self.xx[:, 1] += self.dt*self.g(self.ts[1])
    #     else:
    #         self.xx[:, 1] += self.dt*np.array([self.g[n](self.ts[1]) for n in range(len(self.g))])
    #     for n in range(self.dim):
    #         self.x[n] = sp.CubicSpline(self.ts[:2], self.xx[n][:2])
    #     xx_t2 = self.eval_rhs(self.ts[1])
    #     for n in range(self.dim):
    #         self.xx[n][1] = self.xx[n][0] + self.dt*0.5*(0. + xx_t2[n])
    #     if normalize:
    #         self.xx[:, 1] /= np.sum(self.xx[:, 1])
    #     for n in range(self.dim):
    #         self.x[n] = sp.CubicSpline(self.ts[:2], self.xx[n][:2])
    #
    #     # iterate through all time steps and build trapezoid interpolant and do one Picard iterations
    #     # on the way
    #     for k in range(2, len(self.ts)):
    #         xx_t = self.eval_rhs(self.ts[k - 1])
    #         for n in range(self.dim):
    #             self.xx[n][k] = self.xx[n][k - 1] + self.dt*xx_t[n]
    #             self.x[n] = sp.CubicSpline(self.ts[:(k + 1)], self.xx[n][:(k + 1)])
    #         xx_t2 = self.eval_rhs(self.ts[k])
    #         for n in range(self.dim):
    #             self.xx[n][k] = self.xx[n][k - 1] + self.dt*0.5*(xx_t[n] + xx_t2[n])
    #         if normalize:
    #             self.xx[:, k] /= np.sum(self.xx[:, k])
    #         for n in range(self.dim):
    #             self.x[n] = sp.CubicSpline(self.ts[:(k + 1)], self.xx[n][:(k + 1)])
    #
    #     return (0., 0.)

    def fast_euler_solve(self, tol=10**(-2), max_iter=20, normalize=False):
        # Use a simple multistep backward-Euler scheme to integrate the target function
        # as an integrator, simply use Simpson's rule on the already obtained steps
        # error estimates are not yet available

        self.xx = np.array([np.concatenate((np.array([self.I[0][n], self.I[0][n]]), np.zeros(self.N - 2))) for n in range(self.dim)])
        self.zz = np.array([np.concatenate((np.array([self.I[1][n], self.I[1][n]]), np.zeros(self.N - 2))) for n in range(self.dim)])
        if self.vec_g:
            self.xx[:, 1] += self.dt*self.g(self.ts[1])
        else:
            self.xx[:, 1] += self.dt*np.array([self.g[n](self.ts[1]) for n in range(len(self.g))])
        xx_t2 = self.fast_eval_rhs(1)
        self.xx[:, 1] = self.xx[:, 0] + self.dt*0.5*(0. + xx_t2)
        self.zz[:, 1] = self.fast_eval_rhs(1, no_Q=True)
        if normalize:
            self.xx[:, 1] /= np.sum(self.xx[:, 1])

        # iterate through all time steps and build trapezoid interpolant and do one Picard iterations
        # on the way
        for k in range(2, len(self.ts)):
            xx_t = self.fast_eval_rhs(k - 1)
            self.xx[:, k] = self.xx[:, k - 1] + self.dt*xx_t
            xx_t2 = self.fast_eval_rhs(k)
            self.xx[:, k] = self.xx[:, k - 1] + self.dt*0.5*(xx_t + xx_t2)
            self.zz[:, k] = self.fast_eval_rhs(k, no_Q=True)
            if normalize:
                self.xx[:, k] /= np.sum(self.xx[:, k])

        # build the interpolant at the end
        for n in range(self.dim):
            self.x[n] = sp.CubicSpline(self.ts, self.xx[n])
            self.z[n] = sp.CubicSpline(self.ts, self.zz[n])

        return (0., 0.)

    # def karpel_solve(self, tol=10**(-2), max_iter=20):
    #     # Use a ore-packaged IDE solver (made by Josh Karpel)
    #     # to integrate the target function over self.T
    #     lim = [(lambda t: self.T[0]), (lambda t: self.T[1])]
    #     if self.V[1] == True:
    #         lim[1] = (lambda t: t)
    #     elif self.V[0] == True:
    #         lim[0] = (lambda t: t)
    #     solver = IDESolver(
    #             x = self.ts,
    #             y_0 = self.I,
    #             c = lambda x, y: [self.g[n](x) for n in range(self.dim)],
    #             d = lambda x: self.Q,
    #             k = lambda x, s: [self.K[n](x - s) for n in range(self.dim)],
    #             f = lambda y: y,
    #             lower_bound = lambda x: lim[0](x),
    #             upper_bound = lambda x: lim[1](x),
    #             global_error_tolerance = tol,
    #             max_iterations = max_iter
    #     )
    #     solver.solve()
    #     # extract the node values and build the interpolants
    #     for n in range(self.dim):
    #         for k in range(len(self.ts)):
    #             self.xx[n][k] = solver.y[k][n]
    #         self.x[n] = sp.CubicSpline(ts, self.xx[n])
    #     return (solver.global_error, solver.iteration)

    def eval_rhs(self, t, b=None, no_Q=False):
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
        return self._rhs(t, lim, no_Q)

    def fast_eval_rhs(self, k, b=None, no_Q=False):
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
        return self._fast_rhs(k, lim, no_Q)

    def __call__(self, t, z=False):
        if z:
            return np.array([self.z[n](t) for n in range(self.dim)])
        # else call the function x
        return np.array([self.x[n](t) for n in range(self.dim)])

    def _rhs(self, t, lim, no_Q=False):
        new_xx = np.zeros(self.dim)
        for n in range(self.dim):
            fun = (lambda s: self.K[n](t - s)*self.x[n](s))
            new_xx[n] = si.quad(fun, lim[0], lim[1])[0]
        if self.vec_g:
            gg = self.g(t)
        else:
            gg = np.zeros(self.dim)
            for n in range(self.dim):
                gg[n] = self.g[n](t)
        if no_Q:
            new_xx += gg
        else:
            new_xx = np.dot(self.Q, new_xx)
            new_xx += np.dot(self.Q, gg)
        return new_xx

    def _fast_rhs(self, k, lim, no_Q=False):
        new_xx = np.zeros(self.dim)
        for n in range(self.dim):
            #fun = (lambda s: self.K[n](t - s)*self.x[n](s))
            #new_xx[n] = si.quad(fun, lim[0], lim[1])[0]
            new_xx[n] = si.trapz(self.KK[n][k:0:-1]*self.xx[n][:k], dx=self.dt)
        if self.vec_g:
            gg = self.g(self.ts[k])
        else:
            gg = np.zeros(self.dim)
            for n in range(self.dim):
                gg[n] = self.g[n](self.ts[k])
        if no_Q:
            new_xx += gg
        else:
            new_xx = np.dot(self.Q, new_xx)
            new_xx += np.dot(self.Q, gg)
        return new_xx
