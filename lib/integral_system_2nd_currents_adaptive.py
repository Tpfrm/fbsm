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

class integral_system_2nd_currents_adaptive:
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
        # N is here a variable for the maximal initially reserved space
        self.N = N
        self.uts = np.linspace(self.T[0], self.T[1], self.N)
        self.dt = _diff([self.uts[0], self.uts[1]])
        # sample fine grid solution for compatibility at the end
        self.xx = np.zeros([self.dim, self.N])
        self.zz = np.zeros([self.dim, self.N])
        # pre-allocate adaptive interpolation nodes
        self.xad = np.zeros([self.dim, self.N])
        self.zad = np.zeros([self.dim, self.N])
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
        # create lookup tables. Ensure, all antiderivatives are linearly interpolated
        self.global_ts = np.arange(0., 100., self.dt)
        self.KK = np.zeros([self.dim, len(self.global_ts)])
        self.k = [None for _ in range(self.dim)]
        self.ik = [None for _ in range(self.dim)]
        self.iik = [None for _ in range(self.dim)]
        for n in range(self.dim):
            self.KK[n] = self.K[n](self.global_ts)
            self.k[n] = sp.UnivariateSpline(self.global_ts, self.KK[n], k=1, s=0)
            self.ik[n] = self.k[n].antiderivative()
            self.ik[n] = sp.UnivariateSpline(self.global_ts, self.ik[n](self.global_ts), k=1, s=0)
            self.iik[n] = self.ik[n].antiderivative()
            self.iik[n] = sp.UnivariateSpline(self.global_ts, self.iik[n](self.global_ts), k=1, s=0)
        # Also allocate nodes and an interpolant object for the output currents z
        self.z = [None for _ in range(self.dim)]
        self.p = [None for _ in range(self.dim)]
        # set tolerance for integration error
        self.tol = 10.**(-3)

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
        ##################################################################################
        # set h first as the smallest step
        h = self.dt
        # choose extreme values
        self.h_max = 1.
        self.h_min = self.dt/2
        self.f_max = 3.
        self.f_min = 1./3.
        # preallocate a large array and reduce it down once finished
        self.ts = np.zeros(self.N)
        k = 1
        t = self.T[0]
        self.ts[0] = t
        # initialize currents
        self.zad[:, 0] = self.__g(t)
        self.xad[:, 0] = np.dot(self.Q, self.zad[:, 0])
        # as long as we haven't reached our destination, we proceed
        while not np.isclose(t, self.T[1]):
            # increase t to next step
            if t + h > self.T[1]:
                h = self.T[1] - t
            # set new t
            # first, do one single step
            self.ts[k] = t + h
            xad_one, _ = self._do_a_step(t + h, k, self.xad)
            # then, do two half steps
            self.ts[k] = t + h/2
            self.xad[:, k], _ = self._do_a_step(t + h/2, k, self.xad)
            self.ts[k + 1] = t + h
            xad_half, zad_half = self._do_a_step(t + h, k + 1, self.xad)
            self.xad[:, k] = np.zeros(self.dim)  # reset xad again to previous state
            self.ts[k + 1] = 0.  # reset ts again to previous state
            # adjust step size
            h_new = self._adjust_step_size(h, xad_one, xad_half)
            # if h_new < h, recalculate the current window with the reduced step size and so on
            if h_new >= h:
                # interval accepted. Update t, ts, xad, zad and increase k
                t += h
                self.ts[k] = t
                self.xad[:, k] = xad_half
                self.zad[:, k] = zad_half
                k += 1
            h = h_new
        # shorten all arrays to filled size
        self.xad = self.xad[:, :k]
        self.zad = self.zad[:, :k]
        self.ts = self.ts[:k]
        # interpolate continuous function
        for n in range(self.dim):
            self.x[n] = sp.CubicSpline(self.ts, self.xad[n])
            self.z[n] = sp.CubicSpline(self.ts, self.zad[n])
            self.p[n] = sp.CubicSpline(self.ts, self.xad[n] - self.zad[n]).antiderivative()
            # fill fine grained collocation nodes for compatibility
            self.xx[n, :] = self.x[n](self.uts)
            self.zz[n, :] = self.z[n](self.uts)
        # return nothing
        return (0, 0)

    def _dI(self, t, k, s):
        return (self.iik[s](t - self.ts[k]) - self.iik[s](t - self.ts[k + 1]))/(self.ts[k + 1] - self.ts[k])

    def _D(self, t, k_I):
        return np.array([self._dI(t, k_I, s) for s in range(self.dim)])

    def _F(self, t, k_F):
        return np.array([self.ik[s](t - self.ts[k_F]) for s in range(self.dim)])

    def _do_a_step(self, t, k, xad):
        # unity matrix
        I = np.eye(self.dim)
        # gather all times, we have already calculated, for all states
        current_xs = xad[:, :k]
        # build the D matrix to calculate all scalar products
        D = np.empty_like(current_xs)
        D[:, 0] = self._D(t, 0) - self._F(t, 0)
        for m in range(1, k):
            D[:, m] = self._D(t, m - 1) - self._D(t, m)
        mem = np.sum(D*current_xs, axis=1)
        # build the rhs by multiplying the embedded transition matrix to the left
        rhs_woQ = mem + self.__g(t)
        rhs = np.dot(self.Q, rhs_woQ)
        lhs = I - np.dot(self.Q, np.diag(self._D(t, k - 1) - self._F(t, k)))
        # solve the equation system
        xad_new = nla.solve(lhs, rhs)
        # solve for the output current as well
        zad_new = rhs_woQ + (self._D(t, k - 1) - self._F(t, k))*xad_new
        return (xad_new, zad_new)

    def _adjust_step_size(self, h, x_one, x_half):
        E = np.max(np.absolute(x_one - x_half)) + 10**(-8)  # last guy for numerical stability
        # ensure, the step size does not change abruptly and not to much overall
        f = np.max([np.min([self.tol/E, self.f_max]), self.f_min])
        h_new = np.max([np.min([h*f, self.h_max]), self.h_min])
        #print('E: ' + str(E) + ', h_new: ' + str(h_new) + ', h_old: ' + str(h))
        return h_new


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

    def __g(self, t):
        if self.vec_g:
            return self.g(t)
        else:
            return np.array([self.g[n](t) for n in range(self.dim)])

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
