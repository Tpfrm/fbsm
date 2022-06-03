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
#from lib.maximum_system_2nd_currents import maximum_system_2nd_currents as mes2c
from lib.maximum_system_2nd import maximum_system_2nd as mes2

class viterbi_CTSMC:
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
            self.ss.p_inf = np.hstack(sla.null_space(self.ss.Q.T, rcond=10**(-9)))
            self.ss.p_inf = self.ss.p_inf/np.sum(self.ss.p_inf)
            print(self.ss.p_inf)

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
            self.k.likelihood_factors = []
            self.k.normalization_factors = []
            self.k.normalization_preset = preset_normalization
            self.k.local_time = 0.
            # the forward and backward marginals calculated from the currents
            self.k.p_t = self.p_0
            self.k.b_t = self.p_0
            self.k.callback = step_callback
            self.k.Q = np.transpose(self.jp_dists)
            self.k.Qb = self.jp_dists
            # no need to calculate memory kernels in this setting
            self.k.K = [(lambda t, s=s: self.sjt_dists[s].fun(t)) for s in range(self.num_states)]
            self.k.Kb = self.k.K
            self.k.IK = [(lambda t, s=s: self.sjt_dists[s].fun(t, f='sf')) for s in range(self.num_states)]
            self.k.IKb = self.k.IK
            # initialize the inhomogeneity with a steady state
            if initial == 'forward':
                initial_inhomogeneity = np.transpose(np.multiply(self.ss.linf*self.ss.p_inf, np.ones([N, self.num_states])))
            elif initial == 'backward':
                initial_inhomogeneity = np.transpose(np.multiply(np.ones_like(self.ss.linf)/self.num_states, np.ones([N, self.num_states])))
            self.k.inhomogeneity = [ies2([(lambda t: 0) for _ in range(self.num_states)], self.k.K, np.eye(self.num_states), [-tau_max, 0.], [0, 0], initial_inhomogeneity)]
            self.k.integral_inhomogeneity = [ies2([(lambda t: 0) for _ in range(self.num_states)], self.k.IK, np.eye(self.num_states), [-tau_max, 0.], [0, 0], initial_inhomogeneity)]
            # print(self.k.inhomogeneity[0].fast_eval_rhs(0.))
            # print(self.k.integral_inhomogeneity[0].fast_eval_rhs(0.))
            # for s in range(self.num_states):
            #     print(np.trapz(self.k.IK[s](self.k.ts), dx=self.k.dt))
            # print(np.dot(self.jp_dists, np.ones_like(self.ss.p_inf)))
            # exit(0)
            self._update_likelihood(np.ones_like(self.ss.p_inf))
            if self.k.normalization_preset is None:
                self._update_normalization(1.)
            else:
                self._update_normalization(1.)
            # set the currents
            self.k.phia_t = self.ss.linf*self.ss.p_inf
            self.k.psib_t = np.ones_like(self.ss.linf)/self.num_states
            self.k.psia_t = self.ss.linf*self.ss.p_inf
            self.k.phib_t = np.ones_like(self.ss.linf)/self.num_states
            self.k.current_equation = None
            self.k.current_marginal = None

    def kolmogorov_forward(self, t_delta):
        # at this point, we should already have calculated the requested time
        # interval, so just set to the calculated marginal (with current)
        self.k.p_t, self.k.phia_t, self.k.psia_t = self._advance(self.k.p_t, t_delta)

    def kolmogorov_backward(self, t_delta):
        # at this point, we should already have calculated the requested time
        # interval, so just set to the calculated marginal (with current)
        self.k.b_t, self.k.psib_t, self.k.phib_t = self._recede(self.k.b_t, t_delta)

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
            self.k.current_equation = self._create_local_forward_equation(self.k.local_time, self.k.local_time + t_delta, N)
            # already solve completely for the current interval
            self.k.current_equation.solve()
            # After this, we have the currents we need. So we can setup an integrator to obtain the marginals
            self.k.current_marginal = self._create_local_forward_integrator(self.k.local_time, self.k.local_time + t_delta, self.k.current_equation.xx)
        else:
            # create a local integrodifferential equation for the requested interval
            self.k.current_equation = self._create_local_backward_equation(self.k.local_time, self.k.local_time + t_delta, N)
            # already solve completely for the current interval
            self.k.current_equation.solve()
            # After this, we have the currents we need. So we can setup an integrator to obtain the marginals
            self.k.current_marginal = self._create_local_backward_integrator(self.k.local_time, self.k.local_time + t_delta, self.k.current_equation.xx)

    def kolmogorov_update(self, u, type='forward'):
        if type == 'backward':
            y = np.copy(np.array(u))
            self.k.b_t *= y
            norm = np.sum(self.k.b_t)
            self.k.b_t /= norm
            self._update_likelihood(y)
            if self.k.normalization_preset is None:
                self._update_normalization(norm)
            else:
                self._update_normalization(self.k.normalization_preset.pop(0))
            # add current resolved time interval to global inhomogeneity
            self._update_inhomogeneity(self.k.current_equation, self.k.local_time)
        else:
            y = np.copy(np.array(u))
            self.k.p_t *= y
            norm = np.sum(self.k.p_t)
            self.k.p_t /= norm
            self._update_likelihood(y)
            if self.k.normalization_preset is None:
                self._update_normalization(norm)
            else:
                self._update_normalization(self.k.normalization_preset.pop())
            # add current resolved time interval to global inhomogeneity
            self._update_inhomogeneity(self.k.current_equation, self.k.local_time)

    def _advance(self, p_t, t_delta):
        # at this point, we should already have solved for this interval, so
        # just return the requested time instance
        self.k.p_t = self.k.current_marginal(self.k.local_time + t_delta)
        self.k.phia_t = self.k.current_equation(self.k.local_time + t_delta)
        self.k.psia_t = self.k.current_equation(self.k.local_time + t_delta, z=True)
        #print(self.k.current_equation(self.k.local_time + t_delta))
        self.k.local_time += t_delta
        return (self.k.p_t, self.k.phia_t, self.k.psia_t)

    def _recede(self, b_t, t_delta):
        # at this point, we should already have solved for this interval, so
        # just return the requested time instance
        self.k.b_t = self.k.current_marginal(self.k.local_time + t_delta)
        self.k.psib_t = self.k.current_equation(self.k.local_time + t_delta)
        self.k.phib_t = self.k.current_equation(self.k.local_time + t_delta, z=True)
        self.k.local_time += t_delta
        return (self.k.b_t, self.k.psib_t, self.k.phib_t)

    # def _recede(self, b_t, l, dt, N):
    #     for s in range(N):
    #         self.k.b_t = self.__recede_step(self.k.b_t, self.k.l, dt)
    #     return self.k.b_t

    ### Functions concerned with building the integrodifferential system

    def _create_local_forward_equation(self, T_0, T_1, N):
        # this creates the new integrodifferential equation and initializes
        # the inhomogeneous term as integrals over constant likelihood
        return ies2c((lambda t: self._evaluate_inhomogeneity(t, False, False)), self.k.K, self.k.Q, [T_0, T_1], N, vec_g=True)
        #return ids2([(lambda t: 0) for _ in range(self.num_states)], self.k.K, self.k.Q, [T_0, T_1], [0, 1], I, N)

    def _create_local_backward_equation(self, T_0, T_1, N):
        # this creates the new integrodifferential equation and initializes
        # the inhomogeneous term as integrals over constant likelihood
        return ies2c((lambda t: self._evaluate_inhomogeneity(t, True, False)), self.k.Kb, self.k.Qb, [T_0, T_1], N, vec_g=True)

    def _create_local_forward_integrator(self, T_0, T_1, collocation_nodes):
        # generate the integrator and return its function
        # calling the function at t should return the marginals
        ies = ies2((lambda t: self._evaluate_inhomogeneity(t, False, True)), self.k.IK, np.eye(self.num_states), [T_0, T_1], [0, 1], collocation_nodes, vec_g=True)
        def __marginals(t, i=ies):
            p = i.fast_eval_rhs(t)
            return p/np.sum(p)
        return __marginals

    def _create_local_backward_integrator(self, T_0, T_1, collocation_nodes):
        ies = ies2((lambda t: self._evaluate_inhomogeneity(t, True, True)), self.k.IK, np.eye(self.num_states), [T_0, T_1], [0, 1], collocation_nodes, vec_g=True)
        def __marginals(t, i=ies):
            b = self.ss.linf*i.fast_eval_rhs(t)
            return b/np.sum(b)
        return __marginals

    def _update_inhomogeneity(self, ids, T_1):
        # the old current equation becomes a simple integral over the previous
        # time interval. It is then added to the list of inhomogeneity terms
        #print(np.transpose(ids.xx))
        #self.old_ps.insert(0, np.copy(self.k.p_t))
        #print([ids.p[s](T_1) for s in range(self.num_states)])
        new_interval = ies2([(lambda t: 0) for _ in range(self.num_states)], ids.K, np.eye(self.num_states), [ids.T[0], T_1], [0, 0], ids.xx)
        self.k.inhomogeneity.insert(0, new_interval)
        new_integrator_interval = ies2([(lambda t: 0) for _ in range(self.num_states)], self.k.IK, np.eye(self.num_states), [ids.T[0], T_1], [0, 0], ids.xx)
        self.k.integral_inhomogeneity.insert(0, new_integrator_interval)

    def _update_likelihood(self, factor):
        self.k.likelihood_factors.insert(0, factor)

    def _update_normalization(self, factor):
        self.k.normalization_factors.insert(0, float(factor))

    def _evaluate_inhomogeneity(self, t, backward=False, integral=False):
        # omega contains the partial likelihood product
        likelihood = iter(self.k.likelihood_factors)
        normalizer = iter(self.k.normalization_factors)
        # omega = np.ones(self.num_states)
        factor = np.ones(self.num_states)
        # norm = 1.
        result = np.zeros(self.num_states)
        if backward:
            Q = self.k.Qb
        else:
            Q = self.k.Q
        if integral:
            I = self.k.integral_inhomogeneity
        else:
            I = self.k.inhomogeneity
        for term in I:
            factor *= next(likelihood)/next(normalizer)
            result += np.copy(factor*term.fast_eval_rhs(t))
            #print(result)
            #result += np.copy(np.dot(Q, omega*term.fast_eval_rhs(t)))
            #print(next(old_ps))
            #result += np.copy(np.dot(self.k.Q, omega*next(old_ps)))
        return result

    ### Functions concerned with the viterbi algorithm

    def viterbi_setup(self, tau_max, dt):
        # must run setup before we can forward the marginal
        N = int(tau_max/dt)
        if (self.v is None):
            self.v = _Viterbi()
            self.v.dt = dt
            self.v.tau_max = tau_max
            self.v.ts = np.arange(0, tau_max, dt)
            self.v.likelihood_factors = []
            self.v.Q = np.transpose(self.jp_dists)
            # no need to calculate memory kernels in this setting
            self.v.K = [(lambda t, s=s: self.sjt_dists[s].fun(t)) for s in range(self.num_states)]
            self.v.IK = [(lambda t, s=s: self.sjt_dists[s].fun(t, f='sf')) for s in range(self.num_states)]
            # initialize the inhomogeneity with a steady state
            initial_inhomogeneity = np.transpose(np.multiply(self.ss.linf*self.ss.p_inf, np.ones([N, self.num_states])))
            self.v.inhomogeneity = [mes2([(lambda t: 0) for _ in range(self.num_states)], self.v.K, np.eye(self.num_states), [-tau_max, 0.], [0, 0], initial_inhomogeneity)]
            self.v.endpoint_inhomogeneity = [mes2([(lambda t: 0) for _ in range(self.num_states)], self.v.IK, np.eye(self.num_states), [-tau_max, 0.], [0, 0], initial_inhomogeneity)]
            self.v.psia_t = self.ss.linf*self.ss.p_inf
            self.v.current_window_equation = None
            self.v.endpoint_equation = None

    def viterbi_trace_back(self):
        # we have reached an endpoint. Thus, we need to trace back to the start
        # prepare a list of states and waiting times
        self.v.trajectory = list()
        # create the endpoint equation
        self._create_endpoint_equation(self.v.current_window_equation, self.v.T)
        # now resolve the terminal state
        mx, ix = self.v.endpoint_equation.fast_eval_arg(self.v.T)
        # find the overall maximum
        s_current = np.argmax(mx)
        t_current = self.v.T
        self.v.trajectory.insert(0, tuple(s_current, t_current))
        w_current = ix[s_current, 1]
        t_current = ix[s_current, 2]
        s_current = ix[s_current, 0]
        while (t_current > 0):
            # insert the current values into the list
            self.v.trajectory.insert(0, tuple(s_current, t_current))
            # prune the inhomogeneity so far as to bring the next jump window to front
            if w_current > 0:
                self.v.current_window_equation = self.v.inhomogeneity[w_current - 1]
                self.v.likelihood_factors = self.v.likelihood_factors[w_current:]
                self.v.inhomogeneity = self.v.inhomogeneity[w_current:]
                # set the equation to second kind
                self.v.current_window_equation.V[1] = 1
                # register the inhomogeneity with the new equation
                self.v.current_window_equation.g = (lambda t: self._arg_inhomogeneity(t))
                self.v.current_window_equation.vec_g = True
            mx, ix = self.v.current_window_equation.fast_eval_arg(t_current)
            w_current = ix[s_current, 1]
            t_current = ix[s_current, 2]
            s_current = ix[s_current, 0]
        self.v.trajectory.insert(0, tuple(s_current, 0.))
        return self.v.trajectory

    def viterbi_forward(self, t_delta):
        # at this point, we should already have calculated the requested time
        # interval, so just set to the calculated marginal (with current)
        self.v.phiv_t = self._advance(self.v.phiv_t, t_delta)

    def viterbi_reset(self):
        self.v = None

    def viterbi_notify(self, t_delta):
        if (self.v is None):
            return False
        N = int(t_delta/self.v.dt)
        if (N - t_delta/self.v.dt >= self.v.dt*self.v.dt):
            print("Warning (kolmogorov): non-negligible discretization error at the end!")
        # create a local integrodifferential equation for the requested interval
        self.v.current_window_equation = self._create_current_window_equation(self.v.local_time, self.v.local_time + t_delta, N)
        # already solve completely for the current interval
        self.v.current_window_equation.solve()

    def viterbi_update(self, u):
        y = np.copy(np.array(u))
        self._update_likelihood(y)
        # add current resolved time interval to global inhomogeneity
        self._update_inhomogeneity(self.v.current_window_equation, self.v.local_time)

    def _advance(self, phiv_t, t_delta):
        # at this point, we should already have solved for this interval, so
        # just return the requested time instance
        self.v.phiv_t = self.v.current_window_equation(self.v.local_time + t_delta)
        self.v.local_time += t_delta
        return self.k.phiv_t

    def _create_current_window_equation(self, T_0, T_1, N):
        # this creates the new integrodifferential equation and initializes
        # the inhomogeneous term as integrals over constant likelihood
        return mes2((lambda t: self._evaluate_inhomogeneity(t)), self.v.K, self.v.Q, [T_0, T_1], [0, 1], np.zeros(N), vec_g=True)
        #return ids2([(lambda t: 0) for _ in range(self.num_states)], self.k.K, self.k.Q, [T_0, T_1], [0, 1], I, N)

    def _create_endpoint_equation(self, ids, T_1):
        # this creates the new integrodifferential equation and initializes
        # the inhomogeneous term as integrals over constant likelihood
        return mes2((lambda t: self._evaluate_inhomogeneity(t, endpoint=True)), self.v.IK, np.eye(self.num_states), [ids.T[0], T_1], [0, 0], ids.xx, vec_g=True)
        #return ids2([(lambda t: 0) for _ in range(self.num_states)], self.k.K, self.k.Q, [T_0, T_1], [0, 1], I, N)

    def _update_inhomogeneity(self, ids, T_1):
        # the old current equation becomes a simple maximum over the previous
        # time interval. It is then added to the list of inhomogeneity terms
        new_interval = mes2([(lambda t: 0) for _ in range(self.num_states)], ids.K, np.eye(self.num_states), [ids.T[0], T_1], [0, 0], ids.xx)
        self.v.inhomogeneity.insert(0, new_interval)
        new_endpoint_interval = mes2([(lambda t: 0) for _ in range(self.num_states)], self.v.IK, np.eye(self.num_states), [ids.T[0], T_1], [0, 0], ids.xx)
        self.v.endpoint_inhomogeneity.insert(0, new_endpoint_interval)

    def _update_likelihood(self, factor):
        self.v.likelihood_factors.insert(0, factor)

    def _evaluate_inhomogeneity(self, t, endpoint=False):
        # omega contains the partial likelihood product
        likelihood = iter(self.v.likelihood_factors)
        factor = np.ones(self.num_states)
        result = np.zeros(self.num_states)
        if endpoint:
            I = self.v.endpoint_inhomogeneity
        else:
            I = self.v.inhomogeneity
        res = np.zeros(self.num_states)
        for term in I:
            factor *= next(likelihood)
            # get the maximum value at the current window
            res = np.maximum(res, factor*term.fast_eval_rhs(t))
        return result

    def _arg_inhomogeneity(self, t, endpoint=False):
        # omega contains the partial likelihood product
        likelihood = iter(self.k.likelihood_factors)
        factor = np.ones(self.num_states)
        result = np.zeros([len(self.k.likelihood_factors), self.num_states])
        if endpoint:
            I = self.k.endpoint_inhomogeneity
        else:
            I = self.k.inhomogeneity
        n = 0
        res = np.zeros([len(self.k.likelihood_factors), self.num_states])
        idx = np.zeros([len(self.k.likelihood_factors), self.num_states, 3])
        for term in I:
            factor *= next(likelihood)#/next(normalizer)
            # get the maximum value of the current window
            res[n, :], idx[n, :, :] = term.fast_eval_arg(t)
            # multiply the likelihood modifier to the maxima
            res[n, :] *= factor
            n += 1
        # for each state, find the maximizing window and return the time as well
        for s in range(self.num_states):
            result[s, 1] = np.argmax(res[:, s])
            result[s, 0] = res[result[s, 1], s]
            result[s, 2] = idx[result[s, 1], s, 2]  # at third position, we got the time
        return (result[:, 0], result[:, 1], result[:, 2])


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

class _Viterbi:
    def __init__(self, dt=0, tau_max=0):
        self.dt = dt
        self.tau_max = tau_max

class _Simulation:
    def __init__(self, state=0, sjt=0):
        self.state = state
        self.sjt = sjt
        self.time = 0.
        self.jumps = 0
