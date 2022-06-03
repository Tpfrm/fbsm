# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:28:33 2020

@author: user
"""

from .aux import auxilliary as aux
import numpy.linalg as nla
import scipy.linalg as sla
import scipy.integrate as si
import scipy.interpolate as sp
import numpy as np
import matplotlib.pyplot as plt
import math
from copy import copy, deepcopy
from lib.aux.flt import flt, iflt
from lib.integrodifferential_system_2nd_currents import integrodifferential_system_2nd_currents as ids2
from lib.integral_system_2nd import integral_system_2nd as ies2

class integrodifferential_CTSMC:
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
        self.name = 'integrodifferential CTSMC / integrodifferential fw/bw equations'
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

    def kolmogorov_setup(self, tau_max, dt, step_callback=None, preset_normalization=None, initial='forward'):
        # must run setup before we can forward the marginal
        N = int(tau_max/dt)
        if (self.k is None):
            self.k = _Kolmogorov()
            self.k.dt = dt
            self.k.tau_max = tau_max
            self.k.ts = np.arange(0, tau_max, dt)
            self._calculate_memory_kernels(self.k.ts, self.k.dt)
            self.k.likelihood_factors = []
            self.k.normalization_factors = []
            self.k.normalization_preset = False
            if preset_normalization is not None:
                self.k.normalization_factors = list(preset_normalization)
                self.k.normalization_preset = True
            self.k.local_time = 0.
            self.k.p_t = self.p_0
            self.k.b_t = np.ones(self.num_states)
            self.k.callback = step_callback
            self.k.Q = np.transpose(self.jp_dists) - np.eye(self.num_states)
            self.k.Qb = np.dot(np.diag(self.ss.linf), self.jp_dists - np.eye(self.num_states))
            self.k.inhomogeneity = []
            self.k.current_equation = None
            self.k.psia_t = self.ss.linf*self.ss.p_inf
            self.k.phia_t = np.dot(self.jp_dists, self.k.psia_t)
            self.k.phib_t = self.ss.linf*self.ss.p_inf
            self.k.psib_t = np.dot(self.jp_dists, self.k.phib_t)

    def kolmogorov_forward(self, t_delta):
        # at this point, we should already have calculated the requested time
        # interval, so just return the calculated marginal
        self.k.p_t, self.k.psia_t = self._advance(self.k.p_t, t_delta)
        # also calculate the respective input current
        self.k.phia_t = np.dot(np.transpose(self.jp_dists), self.k.psia_t)

    def kolmogorov_backward(self, t_delta):
        # at this point, we should already have calculated the requested time
        # interval, so just return the calculated marginal
        self.k.b_t, self.k.phib_t = self._recede(self.k.b_t, t_delta)
        # also calculate the respective output current
        self.k.psib_t = np.dot(self.jp_dists, self.k.phib_t)

    # def kolmogorov_backward(self, t_delta):
    #     # call setup at least once before
    #     if (self.k is None):
    #         return False
    #     N = int(t_delta/self.k.dt)
    #     if (N - t_delta/self.k.dt >= self.k.dt*self.k.dt):
    #         print("Warning (kolmogorov): non-negligible discretization error at the end!")
    #     self.k.b_t = self._recede(self.k.b_t, self.k.l, self.k.dt, N)

    def kolmogorov_marginal(self, type='forward'):
        res = None
        if type == 'backward':
            res = np.copy(self.k.b_t)
        else:
            res = np.copy(self.k.p_t)
        return res/np.sum(res)

    def kolmogorov_reset(self):
        self.k = None

    def kolmogorov_notify(self, t_delta, type='forward'):
        if (self.k is None):
            return False
        N = int(t_delta/self.k.dt)
        if (N - t_delta/self.k.dt >= self.k.dt*self.k.dt):
            print("Warning (kolmogorov): non-negligible discretization error at the end!")
        if type == 'forward':
            # create a local integrodifferential equation for the requested interval
            self.k.current_equation = self._create_local_forward_equation([self.k.p_t, self.k.psia_t], self.k.local_time, self.k.local_time + t_delta, N)
            # already solve completely for the current interval
            self.k.current_equation.solve()
        else:
            # create a local integrodifferential equation for the requested interval
            self.k.current_equation = self._create_local_backward_equation([self.k.b_t, self.k.phib_t], self.k.local_time, self.k.local_time + t_delta, N)
            # already solve completely for the current interval
            self.k.current_equation.solve()
        pass

    def kolmogorov_update(self, u, type='forward'):
        if type == 'backward':
            y = np.copy(np.array(u))
            self.k.b_t *= y
            norm = np.sum(self.k.b_t)
            self.k.b_t /= norm
            self._update_likelihood(y)
            if not self.k.normalization_preset:
                self._update_normalization(norm)
            # add current resolved time interval to global inhomogeneity
            self._update_inhomogeneity(self.k.current_equation, self.k.local_time)
        else:
            y = np.copy(np.array(u))
            self.k.p_t *= y
            norm = np.sum(self.k.p_t)
            self.k.p_t /= norm
            self._update_likelihood(y)
            if not self.k.normalization_preset:
                self._update_normalization(norm)
            # add current resolved time interval to global inhomogeneity
            self._update_inhomogeneity(self.k.current_equation, self.k.local_time)

    def _advance(self, p_t, t_delta):
        # at this point, we should already have solved for this interval, so
        # just return the requested time instance
        self.k.p_t = self.k.current_equation(self.k.local_time + t_delta)
        self.k.psia_t = self.k.current_equation(self.k.local_time + t_delta, z=True)
        self.k.local_time += t_delta
        return (self.k.p_t, self.k.psia_t)

    def _recede(self, b_t, t_delta):
        # at this point, we should already have solved for this interval, so
        # just return the requested time instance
        self.k.b_t = self.k.current_equation(self.k.local_time + t_delta)
        self.k.phib_t = self.k.current_equation(self.k.local_time + t_delta, z=True)
        self.k.local_time += t_delta
        return (self.k.b_t, self.k.phib_t)

    # def _recede(self, b_t, l, dt, N):
    #     for s in range(N):
    #         self.k.b_t = self.__recede_step(self.k.b_t, self.k.l, dt)
    #     return self.k.b_t

    ### Functions concerned with building the integrodifferential system

    def _calculate_memory_kernels(self, ts, dt):
        # use the fast Laplace transform for calculation and then
        # use cubic splines to create the continuous kernel instance
        self.k.K = [None for _ in range(self.num_states)]
        for n in range(self.num_states):
            cp = flt(self.sjt_dists[n].fun(ts, f='pdf'), ts)
            cs = flt(self.sjt_dists[n].fun(ts, f='sf'), ts)
            ck = cp/cs
            kk = iflt(ck, ts)
            kk *= self.ss.linf[n]/np.trapz(kk*dt)
            self.k.K[n] = sp.CubicSpline(ts, kk)
        self.k.Kb = [None for _ in range(self.num_states)]
        for n in range(self.num_states):
            self.k.Kb[n] = (lambda t, n=n: (1./self.ss.linf[n])*self.k.K[n](t))
        if False:
            fig = plt.figure()
            axs = fig.subplots(1, 2)
            axs[0].plot(ts, np.transpose([self.sjt_dists[n].fun(ts) for n in range(self.num_states)]))
            #axs[1].plot(ts, np.transpose([self.k.K[n](ts) for n in range(self.num_states)]))
            axs[1].plot(ts, np.transpose([self.k.Kb[n](ts) for n in range(self.num_states)]))
            plt.show()

    def _create_local_forward_equation(self, I, T_0, T_1, N):
        # this creates the new integrodifferential equation and initializes
        # the inhomogeneous term as integrals over constant likelihood
        return ids2((lambda t: self._evaluate_inhomogeneity(t)), self.k.K, self.k.Q, [T_0, T_1], [0, 1], I, N, vec_g=True)
        #return ids2([(lambda t: 0) for _ in range(self.num_states)], self.k.K, self.k.Q, [T_0, T_1], [0, 1], I, N)

    def _create_local_backward_equation(self, I, T_0, T_1, N):
        # this creates the new integrodifferential equation and initializes
        # the inhomogeneous term as integrals over constant likelihood
        return ids2((lambda t: self._evaluate_inhomogeneity(t, True)), self.k.Kb, self.k.Qb, [T_0, T_1], [0, 1], I, N, vec_g=True)

    def _update_inhomogeneity(self, ids, T_1):
        # the old current equation becomes a simple integral over the previous
        # time interval. It is then added to the list of inhomogeneity terms
        #print(np.transpose(ids.xx))
        #self.old_ps.insert(0, np.copy(self.k.p_t))
        new_interval = ies2([(lambda t: 0) for _ in range(self.num_states)], ids.K, np.eye(self.num_states), [ids.T[0], T_1], [0, 0], ids.xx)
        self.k.inhomogeneity.insert(0, new_interval)

    def _update_likelihood(self, factor):
        self.k.likelihood_factors.insert(0, factor)

    def _update_normalization(self, factor):
        self.k.normalization_factors.insert(0, float(factor))

    def _evaluate_inhomogeneity(self, t, backward=False):
        # omega contains the partial likelihood product
        likelihood = iter(self.k.likelihood_factors)
        normalizer = iter(self.k.normalization_factors)
        factor = np.ones(self.num_states)
        result = np.zeros(self.num_states)
        # if backward:
        #     Q = self.k.Qb
        # else:
        #     Q = self.k.Q
        #print('called _evaluate_inhomogeneity')
        for term in self.k.inhomogeneity:
            factor *= next(likelihood)/next(normalizer)
            result += np.copy(factor*term.fast_eval_rhs(t))
            #print(next(old_ps))
            #result += np.copy(np.dot(self.k.Q, omega*next(old_ps)))
        return result

    ############### Section concerned with trajectory simulation

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
