# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:28:33 2020

@author: user
"""

from .aux import auxilliary as aux
import scipy.linalg as sla
import numpy.linalg as nla
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy

class Discretized_CTSMC:
    # the constructor of the SMCTBN expects a set of SMCTBNNodes, an Adjacency matrix, a randomizer dictionary
    def __init__(self, initial_state, randomizer=None):
        self.p_0 = initial_state
        self.num_states = len(initial_state)
        self.p_c = 'steady'
        self.randomizer = randomizer
        self.sjt_dists = [None for _ in range(self.num_states)]
        self.lf = [None for _ in range(self.num_states)]
        self.jp_dists = np.zeros([self.num_states, self.num_states])
        self.k = None
        self.ss = None
        self.sim = None
        self.name = 'uniformly discretized CTSMC / rectangular sum integration'
        if randomizer is not None:
            self.finalize()

    ############### Initialization functions. Call finalize, once you have configured your SMCTBN

    def finalize(self):
        self._finalize()
        self._steady_state_calculate()

    def _finalize(self):
        if self.randomizer is not None:
            self._randomize()
        self.randomizer = None

    def _randomize(self):
        for s in range(0, self.num_states):
            self.set_sjt_dist(s, self.randomizer['sjt']())
            self.set_jp_dist(s, self.randomizer['jp'](self.num_states - 1, s))
        # now, every state should have an associated sojourn time distribution and jump probabilities to all feasible next states

    def set_sjt_dist(self, s, dist):
        self.sjt_dists[s] = dist
        self.lf[s] = (lambda t: self.sjt_dists[s].fun(t)/self.sjt_dists[s].fun(t, f='sf'))

    def set_jp_dist(self, s, categorical):
        self.jp_dists[s] = categorical

    ############### Section concerned with the master equation

    ### Functions concerned with the steady state

    def _steady_state_calculate(self):
        if (self.ss is None):
            self.ss = _Steady()
            self.ss.linf = np.zeros(self.num_states)
            self.ss.Q = np.zeros([self.num_states, self.num_states])
            self.ss.p_clock_inf = [None for _ in range(self.num_states)]
            for s in range(self.num_states):
                self.ss.linf[s] = 1./self.sjt_dists[s].fun(1, f='moment')
                self.ss.p_clock_inf[s] = (lambda t: self.ss.linf[s]*self.sjt_dists[s].fun(1, f='sf'))
                self.ss.Q[s] = self.jp_dists[s]*self.ss.linf[s]
                self.ss.Q[s][s] = -self.ss.linf[s]
            self.ss.p_inf = np.hstack(sla.null_space(self.ss.Q.T, rcond=10**(-6)))
            self.ss.p_inf = self.ss.p_inf/np.sum(self.ss.p_inf)

    def steady_state_print(self):
        print("--- Steady State Information ---")
        print("Steady state distribution p_inf:")
        print(self.ss.p_inf)
        print("Asymptotic Markov generator Q^T:")
        print(self.ss.Q.T)

    ### Functions concerned with the master equation

    def kolmogorov_setup(self, tau_max, dt, step_callback=None):
        # must run setup before we can forward the marginal
        N = int(tau_max/dt)
        if (self.k is None):
            self.k = _Kolmogorov()
            self.k.p_t = np.zeros([self.num_states, N])
            self.k.l = np.zeros([self.num_states, N])
            self.k.ss_l = np.zeros([self.num_states, N])
            self.k.ss_pc = np.zeros([self.num_states, N])
            self.k.b_t = np.zeros([self.num_states, N])
            self.k.b_norm = np.zeros([self.num_states, N])
            self.k.dt = dt
            self.k.tau_max = tau_max
            self.k.n_steps = N
            self.k.ts = np.linspace(0, tau_max, self.k.n_steps)
            ts = self.k.ts
            for s in range(self.num_states):
                self.k.p_t[s] = self.sjt_dists[s].fun(ts, f='sf')#np.zeros_like(ts) #self.sjt_dists[s].fun(ts, f='sf')#self.sjt_dists[s].fun(ts, f='sf')#np.zeros_like(ts) #self.sjt_dists[s].fun(ts, f='sf')
                #self.k.p_t[s][0] = 1
                self.k.p_t[s] = (self.k.p_t[s]/(np.sum(self.k.p_t[s])*dt))*self.p_0[s]
                self.k.l[s] = self.lf[s](ts)
                self.k.ss_l[s] = self.lf[s](ts)
                self.k.ss_pc[s] = np.copy(self.k.p_t[s]/self.p_0[s])
                # exchange this with np.ones_like(ts)/len(ts)
                self.k.b_t[s] = self.k.l[s]#*self.ss.linf[s]
                self.k.b_t[s] = (self.k.b_t[s]/(np.sum(self.k.b_t[s])*dt))*(self.p_0[s])
                self.k.b_norm[s] = self.sjt_dists[s].fun(self.k.ts, f='sf')*self.ss.linf[s]
            self.k.callback = step_callback

    def kolmogorov_forward(self, t_delta):
        # call setup at least once before
        if (self.k is None):
            return False
        N = int(t_delta/self.k.dt)
        if (N - t_delta/self.k.dt >= self.k.dt*self.k.dt):
            print("Warning (kolmogorov): non-negligible discretization error at the end!")
        self.k.p_t = self._advance(self.k.p_t, self.k.l, self.k.dt, N)

    def kolmogorov_backward(self, t_delta):
        # call setup at least once before
        if (self.k is None):
            return False
        N = int(t_delta/self.k.dt)
        if (N - t_delta/self.k.dt >= self.k.dt*self.k.dt):
            print("Warning (kolmogorov): non-negligible discretization error at the end!")
        self.k.b_t = self._recede(self.k.b_t, self.k.l, self.k.dt, N)

    def kolmogorov_marginal(self, type='forward'):
        res = None
        if type == 'backward':
            res = np.sum(self.k.b_t*self.k.b_norm, axis=1)*self.k.dt
            return res/np.sum(res)
        else:
            res = np.sum(self.k.p_t, axis=1)*self.k.dt
            return res/np.sum(res)

    def kolmogorov_reset(self):
        self.k = None

    def kolmogorov_notify(self, t_delta, type='forward'):
        pass

    def kolmogorov_update(self, u, type='forward'):
        if type == 'backward':
            for s in range(self.num_states):
                self.k.b_t[s] = u[s]*self.k.b_t[s]
                #self.k.b_t[s] = (self.k.b_t[s]/(np.sum(self.k.b_t[s])*self.k.dt))*u[s]*self.k.b_t[s]
            self.k.b_t /= np.sum(self.k.b_t)*self.k.dt
        else:
            for s in range(self.num_states):
                #self.k.p_t[s] = (self.k.p_t[s]/(np.sum(self.k.p_t[s])*self.k.dt))*u[s]*self.k.p_t[s]
                self.k.p_t[s] = u[s]*self.k.p_t[s]
            self.k.p_t /= np.sum(self.k.p_t)*self.k.dt

    def __advance_step(self, p_t, l, dt):
        q_t = np.copy(p_t)
        r_t = l*p_t
        for s in range(self.num_states):
            q_t[s] = Discretized_CTSMC.___shift(p_t[s] - r_t[s]*dt, 1)
            for ss in range(self.num_states):
                if (ss != s):
                    q_t[s][0] += self.jp_dists[ss][s]*np.sum(r_t[ss])*dt
        if self.k.callback is not None:
            self.k.callback(self, q_t, r_t)
        return q_t

    def _advance(self, p_t, l, dt, N):
        for s in range(N):
            self.k.p_t = self.__advance_step(self.k.p_t, self.k.l, dt)
        return self.k.p_t

    def __recede_step(self, b_t, l, dt):
        d_t = np.copy(b_t)
        rst = np.zeros(self.num_states)
        for s in range(self.num_states):
            for ss in range(self.num_states):
                if (ss != s):
                    rst[s] += self.jp_dists[s][ss]*b_t[ss][0]
            d_t[s] = Discretized_CTSMC.___shift(b_t[s], -1) + l[s]*rst[s]*dt
        d_t -= l*d_t*dt
        d_t /= np.sum(np.sum(d_t))
        if self.k.callback is not None:
            self.k.callback(self, d_t, np.array([b_t[s][0] for _ in range(self.num_states)]))
        return d_t

    def _recede(self, b_t, l, dt, N):
        for s in range(N):
            self.k.b_t = self.__recede_step(self.k.b_t, self.k.l, dt)
        return self.k.b_t

    ### Functions concerned with trajectory simulation

    def sim_init(self, s, sjt=0):
        self.sim.state = s
        self.sim.sjt = sjt

    def sim_run(self, end, endby='time', history=0):
        self.sim = _Simulation()
        self.sim_init(aux.sample_categorical(self.p_0.tolist()), history)
        if (endby == 'time'):
            T_end = end
            J_end = np.inf
        else:
            T_end = np.inf
            J_end = math.floor(end)
        # now, do the simulation
        trajectory = [dict(state=copy(self.sim.state), time=self.sim.time)]
        while (self.sim.time < T_end) and (self.sim.jumps < J_end):
            Dt = self.sim_draw_sjt()
            self.sim.time += Dt
            if (self.sim.time >= T_end):
                trajectory += [dict(state=copy(trajectory[-1]['state']), time=T_end)]
                break
            self.sim.state = self.sim_draw_jp()
            self.sim_update(self.sim.state)
            trajectory += [dict(state=copy(self.sim.state), time=self.sim.time)]
            self.sim.jumps += 1
        return trajectory

    def sim_draw_sjt(self):
        rv = 0
        rv = self.sjt_dists[self.sim.state].draw_truncated(self.sim.sjt)
        return (rv - self.sim.sjt)

    def sim_draw_jp(self):
        nxt = aux.sample_categorical(self.jp_dists[self.sim.state].tolist())
        return nxt

    def sim_update(self, s):
        self.sim.sjt = 0
        self.sim.state = s

    ### Helper functions for performance/convenience, no essential semantic

    def ___shift(xs, n):
        e = np.empty_like(xs)
        if n >= 0:
            e[:n] = 0.
            e[n:] = xs[:-n]
        else:
            e[n:] = 0.
            e[:n] = xs[-n:]
        return e

### Helper classes from this module

class _Steady:
    def __init__(self, p_inf=0, linf=0, Q=0, p_clock_inf=0):
        self.p_inf = p_inf
        self.linf = linf
        self.Q = Q
        self.p_clock_inf = p_clock_inf

class _Kolmogorov:
    def __init__(self, p_t=0, l=0, dt=0, tau_max=0, n_steps=0):
        self.p_t = p_t
        self.l = l
        self.dt = dt
        self.tau_max = tau_max
        self.n_steps = n_steps
        # temp
        self.ss_l = None
        self.ss_pc = None
        self.ts = None

class _Simulation:
    def __init__(self, state=0, sjt=0):
        self.state = state
        self.sjt = sjt
        self.time = 0.
        self.jumps = 0
