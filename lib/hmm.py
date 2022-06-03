# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:28:33 2020

@author: user
"""

import scipy.linalg as sla
import numpy.linalg as nla
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
import lib.aux.auxilliary as aux
from copy import copy, deepcopy

def plot_clock(ctsmc):
    plt.figure()
    plt.plot(ctsmc.k.p_t.T[:250])
    print(ctsmc.k.p_t)
    plt.show()

def dprint(msg):
    #print(msg)
    pass

# This HMM class is a generic implementation.
# It implements filtering/smoothing for any latent stochastic process
# with Markov property and instantaneous observation model

# The latent process must implement a function "kolmogorov_forward" which
# expects a time, and the observation model must implement a compatible
# likelihood function "observation_likelihood"

class HMM:
    # the constructor of the HMM
    # set Dt = np.infty if we solve from observation to observation
    def __init__(self, latent, observation_model, t_0, T, Dt):
        self.latent = latent
        self.om = observation_model
        self.t_0 = t_0
        self.T = T
        self.observations = []
        self.next_obs = -1
        self.prev_obs = -1
        self.Dt = Dt
        self.log_grid = np.round(np.arange(t_0, T + 0.1*Dt, Dt), decimals=5)
        self.next_log = -1
        self.prev_log = -1
        self.t = t_0

    def observe(self, obs):
        for n in range(len(obs)):
            if (obs[n][0] < self.t):
                print("Error (observe): observations lie in the past, which is currently not supported.")
                return False
        if (len(self.observations) != 0):
            self.observations = np.array(np.vstack(self.observations, obs))
        else:
            self.observations = np.array(obs)
        return True

    def set_time(self, t, stick_to_grid=False):
        if t < self.t_0 or t > self.T:
            print("Error (set_time): time set outside specified time window " + str([self.t_0, self.T]) + ", which is currently not supported.")
            return False
        self.t = t
        # calculate the surrounding log points
        # if we are exactly at a grid point, accept it in neither direction
        off = int(np.round((self.t - self.t_0)/self.Dt, decimals=9))
        at_grid = np.isclose((self.t - self.t_0)/self.Dt - off, 0)
        self.prev_log = off
        if at_grid:
            self.next_log = off
        else:
            self.next_log = off + 1
        if stick_to_grid:
            self.t = self.log_grid[self.prev_log]
        # traverse observations to correctly set next and last observation indices
        self.prev_obs = -1
        self.next_obs = -1
        for n in range(len(self.observations)):
            if self.observations[n][0] > self.t:
                self.next_obs = n
                break
            self.prev_obs = n
        dprint('Current time: ' + str(self.t) + '. Observation indices: ' + str([self.prev_obs, self.next_obs]))
        return True

    def viterbi_pseudodiscrete(self, I, p_0, setup_tuple):
        # number of transitions to try. Set -1 to adaptive
        # adaptive means that it crosses b times the overall median is now again below
        # doesn't support multimodal likelihoods if that is even possible in this case
        # this is a bit of a hacky implementation but proofs the concept
        N = -1
        if N > -1:
            print('Fixed step truncation number: ' + str(N))
            # set first phi as None
            phi = [None for _ in range(N + 1)]
            p_T = np.zeros([N, self.latent.num_states])
            lh = np.zeros(N)
            print('Viterbi LH step: ' + str(0))
            p_T[0, :], phi[1] = self._viterbi_pseudodiscrete_outer_loop(None, I, p_0, setup_tuple)
            lh[0] = np.sum(p_T[0, :])
            for n in range(1, N):
                print('Viterbi LH step: ' + str(n))
                p_T[n, :], phi[n + 1] = self._viterbi_pseudodiscrete_outer_loop(phi[n], I, p_0, setup_tuple)
                lh[n] = np.sum(p_T[n, :])
        else:
            print('Adaptive mode')
            # set first phi as None
            phi = [0., 0.]
            p_T = [np.zeros(self.latent.num_states)]
            lh = []
            factor = 10**(-5)
            print('Viterbi LH step: ' + str(0))
            p_T[0], phi[1] = self._viterbi_pseudodiscrete_outer_loop(None, I, p_0, setup_tuple)
            lh += [np.sum(p_T[0])]
            max = lh[-1]
            N = 1
            tol = np.inf
            while tol > factor:
                print('Viterbi LH step: ' + str(N) + ', current score = ' + str(tol))
                p_T_tmp, phi_tmp = self._viterbi_pseudodiscrete_outer_loop(phi[N], I, p_0, setup_tuple)
                p_T += [p_T_tmp]
                phi += [phi_tmp]
                lh += [np.sum(p_T[-1])]
                max = np.max([lh[-1], max])
                if max > 0.:
                    tol = lh[-1]/max
                N += 1
            p_T = np.array(p_T)
            lh = np.array(lh)
            print('Tolerance reached. Chosen truncation number: ' + str(N) + '. Final tolerance: ' + str(tol))
        # also set phi_0
        phi[0] = np.zeros_like(phi[1])
        phi[0][:, 0] = p_0
        # multiply lh with p_T and look for maximum
        p_T_scaled = np.zeros([N, self.latent.num_states])
        for n in range(N):
            p_T_scaled[n, :] = p_T[n, :]*lh[n]
        idx = list()
        idx.append(np.unravel_index(np.argmax(p_T_scaled), p_T_scaled.shape))
        p_T_second = np.copy(p_T_scaled)
        p_T_second[idx[0]] = 0.
        idx.append(np.unravel_index(np.argmax(p_T_second), p_T_scaled.shape))
        all_states = list()
        all_times = list()
        for k in range(len(idx)):
            n_max = idx[k][0]
            s_max = idx[k][1]
            # now we have identified the maximum. Thus, we trace backwards
            M = len(phi[n_max][0, :])
            t_idx = len(phi[n_max][0, :]) - 1
            ts = np.arange(0, len(phi[n_max][0, :]))*self.latent.k.dt
            f = np.zeros([self.latent.num_states, len(phi[n_max][0, :])])
            L = np.zeros([self.latent.num_states, len(phi[n_max][0, :])])
            t = np.zeros([self.latent.num_states, self.latent.num_states])
            for s in range(self.latent.num_states):
                f[s, :] = self.latent.sjt_dists[s].fun(ts)
                L[s, :] = self.latent.sjt_dists[s].fun(ts, f='sf')
                t[s, :] = self.latent.jp_dists[s, :]
            phi_flipped = [None for _ in range(N)]
            for n in range(N):
                phi_flipped[n] = np.flip(phi[n], axis=1)
            # for n in reversed(range(n_max + 1)):
            #     plt.figure()
            #     plt.plot(ts, np.transpose(phi[n]))
            #     plt.show()
            # get the likelihood values
            obs_times = self.observations[:, 0]
            obs_lhs = np.empty([len(obs_times), len(p_0)])
            for n in range(len(obs_times)):
                obs_lhs[n, :] = self.om.observation_likelihood(self.observations[n, 1])
            # define a likelihood-by-path function
            def lh(t_l, t_h):
                b = (obs_times > t_l) & (obs_times < t_h)
                return np.prod(obs_lhs[b], axis=0)
            # create a likelihood matrix
            Lh = np.zeros([self.latent.num_states, M, M])
            for n in range(M):
                for k in range(n, M):
                    t_l = ts[n] + 0.00001
                    t_h = ts[k] - 0.00001
                    Lh[:, n, k] = lh(t_l, t_h)
            states = np.zeros(n_max + 1, dtype=int)
            states[n_max] = s_max
            times = np.zeros(n_max + 1)
            times[n_max] = I[1]
            idxs = np.unravel_index(np.argmax(L[s_max, :]*phi_flipped[n_max - 1][s_max, :]*np.flip(Lh[s_max, :, -1])), L[s_max, :].shape)
            states[n_max - 1] = s_max
            times[n_max - 1] = ts[-idxs[0] - 1]
            off = idxs[0]
            uoff = (len(phi[n_max][0, :]) - off)
            for n in reversed(range(1, n_max - 1)):
                idxs = np.unravel_index(np.argmax(phi_flipped[n][:, off:]*f[:, :uoff]*np.flip(Lh[:, :uoff, uoff], axis=1)*t[:, states[n + 1], None]), f[:, :uoff].shape)
                off += idxs[1]
                states[n] = idxs[0]
                times[n] = ts[uoff - idxs[1]]
                uoff -= idxs[1]
            all_states.append(states)
            all_times.append(times)
        return (all_states, all_times, np.sum(p_T, axis=1))

    def _viterbi_pseudodiscrete_outer_loop(self, phi, I, p_0, setup_tuple):
        # calculate the forward-backward posterior latent state probabilities
        total_duration = I[1] - I[0]
        # initialize the latent chain for the forward pass
        self.latent.p_0 = p_0
        self.latent.ss = None
        self.latent._steady_state_calculate()
        self.latent.kolmogorov_reset()
        self.latent.kolmogorov_setup(*setup_tuple, phis=phi)
        # First, do the forward pass
        self.set_time(I[0], stick_to_grid=True)
        (p_T, phi_next) = self._viterbi_pseudodiscrete_inner_loop(T=total_duration)
        return (p_T, phi_next)

    def _viterbi_pseudodiscrete_inner_loop(self, T=None):
        # TODO: observations can be missed if they lie inconvenient between [t, t + Dt] and np.close()
        if T == None:
            T = self.T - self.t_0
        if self.t_0 + T > self.T:
            print("Error (forward_pass): time set outside specified time window " + str([self.t_0, self.T]) + ", which is currently not supported.")
            return False
        K = int(T/self.Dt) + 1 + 2*len(self.observations)
        dprint(self.Dt)
        dprint(K)
        dprint(self.observations)
        dprint(self.t)
        if self.next_obs != -1 and self.next_obs < len(self.observations):
            self.latent.kolmogorov_notify(self.observations[self.next_obs][0] - self.t)
        else:
            self.latent.kolmogorov_notify(self.T - self.t)
        k = 0
        while(self.t < self.t_0 + T):
            dprint(self.t)
            if (self.next_obs < len(self.observations)):
                t_next = self.observations[self.next_obs][0]
                if (np.isclose(self.t, t_next)):
                    self.t = t_next
                    y = self.observations[self.next_obs][1]
                    self.latent.kolmogorov_update(self.om.observation_likelihood(y))  # !!!!!!! exchanged
                    self.next_obs += 1
                    # doesn't have to be implemented but sometimes allows efficient memory allocation
                    if self.next_obs < len(self.observations):
                        self.latent.kolmogorov_notify(self.observations[self.next_obs][0] - self.t)
                    else:
                        self.latent.kolmogorov_notify(self.T - self.t)
                elif (self.t < t_next):
                    if (self.log_grid[self.next_log] > t_next):
                        self.latent.kolmogorov_forward(t_next - self.t)
                        self.t = np.round(t_next, decimals=5)
                    else:
                        if not np.isclose(self.log_grid[self.next_log] - self.t, 0.):
                            self.latent.kolmogorov_forward(self.log_grid[self.next_log] - self.t)
                        self.t = np.round(self.log_grid[self.next_log], decimals=5)
                        self.next_log += 1
                else:
                    print("Error (forward_pass): non-'causal' event. Aborting.")
                    return False
            else:
                self.latent.kolmogorov_forward(self.log_grid[self.next_log] - self.t)
                self.t = np.round(self.log_grid[self.next_log], decimals=5)
                self.next_log += 1
            #ps['time'][k] = self.t
            #ps['marginal'][k] = self.latent.kolmogorov_marginal()
            #print(ps['time'][k])
            #print(ps['marginal'][k])
            # if k%20 == 0:
            #     print("- step = " + str(k))
            k += 1
        p_T = self.latent.kolmogorov_marginal()
        next_phis = np.copy(self.latent.k.next_phis)
        return (p_T, next_phis)

    def forward_pass(self, T=None, _posterior=0, grid=False):
        # TODO: observations can be missed if they lie inconvenient between [t, t + Dt] and np.close()
        if T == None:
            T = self.T - self.t_0
        if self.t_0 + T > self.T:
            print("Error (forward_pass): time set outside specified time window " + str([self.t_0, self.T]) + ", which is currently not supported.")
            return False
        K = int(T/self.Dt) + 1 + 2*len(self.observations)
        dprint(self.Dt)
        dprint(K)
        dprint(self.observations)
        dprint(self.t)
        if _posterior == 1:
            ps = dict(time=-np.ones(K), marginal=np.zeros([K, self.latent.num_states]), full=np.zeros([K, self.latent.num_states, self.latent.k.n_steps]))
            #ps['full'][0] = np.copy(self.latent.k.p_t)
        elif _posterior == 2 or _posterior == 3:
            ps = dict(time=-np.ones(K), marginal=np.zeros([K, self.latent.num_states]), current={'output': np.zeros([K, self.latent.num_states]), 'input': np.zeros([K, self.latent.num_states])}, normalizers=None)
        else:
            if not grid:
                ps = dict(time=-np.ones(K), marginal=np.zeros([K, self.latent.num_states]))
            else:
                ps = dict(time=-np.ones(K), marginal=np.zeros([K, self.latent.num_states]), grid=list())
        #ps['time'][0] = self.t
        #ps['marginal'][0] = self.latent.kolmogorov_marginal()
        if self.next_obs != -1 and self.next_obs < len(self.observations):
            self.latent.kolmogorov_notify(self.observations[self.next_obs][0] - self.t)
        else:
            self.latent.kolmogorov_notify(self.T - self.t)
        if grid:  # the time-grid has internally already been calculated
            ps['grid'].append(np.copy(self.latent.k.current_equation.ts))
            print(ps['grid'])
        k = 0
        while(self.t < self.t_0 + T):
            dprint(self.t)
            if (self.next_obs < len(self.observations)):
                t_next = self.observations[self.next_obs][0]
                if (np.isclose(self.t, t_next)):
                    self.t = t_next
                    y = self.observations[self.next_obs][1]
                    p_u = self.latent.kolmogorov_marginal()*self.om.observation_likelihood(y)
                    p_u = np.copy(p_u/np.sum(p_u))
                    dprint(p_u)
                    self.latent.kolmogorov_update(self.om.observation_likelihood(y))  # !!!!!!! exchanged
                    self.next_obs += 1
                    # doesn't have to be implemented but sometimes allows efficient memory allocation
                    if self.next_obs < len(self.observations):
                        self.latent.kolmogorov_notify(self.observations[self.next_obs][0] - self.t)
                    else:
                        self.latent.kolmogorov_notify(self.T - self.t)
                    if grid:  # the time-grid has internally already been calculated
                        ps['grid'].append(np.copy(self.latent.k.current_equation.ts))
                        print(ps['grid'])
                elif (self.t < t_next):
                    if (self.log_grid[self.next_log] > t_next):
                        self.latent.kolmogorov_forward(t_next - self.t)
                        self.t = np.round(t_next, decimals=5)
                    else:
                        if not np.isclose(self.log_grid[self.next_log] - self.t, 0.):
                            self.latent.kolmogorov_forward(self.log_grid[self.next_log] - self.t)
                        self.t = np.round(self.log_grid[self.next_log], decimals=5)
                        self.next_log += 1
                else:
                    print("Error (forward_pass): non-'causal' event. Aborting.")
                    return False
            else:
                self.latent.kolmogorov_forward(self.log_grid[self.next_log] - self.t)
                self.t = np.round(self.log_grid[self.next_log], decimals=5)
                self.next_log += 1
            ps['time'][k] = self.t
            ps['marginal'][k] = self.latent.kolmogorov_marginal()
            #print(ps['time'][k])
            #print(ps['marginal'][k])
            if _posterior == 1:
                ps['full'][k] = np.copy(self.latent.k.p_t)
            elif _posterior == 2 or _posterior == 3:
                ps['current']['input'][k] = self.latent.k.phia_t
                ps['current']['output'][k] = self.latent.k.psia_t
            if k%20 == 0:
                print("- step = " + str(k))
            k += 1
        # Also save the normalizers
        if _posterior == 2 or _posterior == 3:
            ps['normalizers'] = np.array(self.latent.k.normalization_factors)
        # finally, delete trailing illegal times bc. of overallocation
        for n in range(len(ps['time']) - len(self.observations), len(ps['time'])):
            if (np.isclose(ps['time'][n], -1.)):
                ps['time'] = ps['time'][:n]
                ps['marginal'] = ps['marginal'][:n]
                if _posterior == 1:
                    ps['full'] = ps['full'][:n]
                elif _posterior == 2 or _posterior == 3:
                    ps['current']['input'] = ps['current']['input'][:n]
                    ps['current']['output'] = ps['current']['output'][:n]
                break
        return ps

    def backward_pass(self, T=None, _posterior=0):
        # TODO: observations can be missed if they lie inconvenient between [t, t + Dt] and np.close()
        if T == None:
            T = self.T - self.t_0
        if self.T - T < self.t_0:
            print("Error (backward_pass): time set outside specified time window " + str([self.t_0, self.T]) + ", which is currently not supported.")
            return False
        K = int(T/self.Dt) + 2 + 2*len(self.observations)
        dprint(self.Dt)
        dprint(K)
        dprint(self.observations)
        dprint(self.t)
        if _posterior == 1:
            bs = dict(time=-np.ones(K), marginal=np.zeros([K, self.latent.num_states]), full=np.zeros([K, self.latent.num_states, self.latent.k.n_steps]))
            #bs['full'][0] = np.copy(self.latent.k.b_t)
        elif _posterior == 2 or _posterior == 3:
            bs = dict(time=-np.ones(K), marginal=np.zeros([K, self.latent.num_states]), current={'output': np.zeros([K, self.latent.num_states]), 'input': np.zeros([K, self.latent.num_states])})
        else:
            bs = dict(time=-np.ones(K), marginal=np.zeros([K, self.latent.num_states]))
        #bs['time'][0] = self.t
        #bs['marginal'][0] = self.latent.kolmogorov_marginal(type='backward')
        if self.prev_obs > -1:
            self.latent.kolmogorov_notify(self.t - self.observations[self.prev_obs][0], type='backward')
        else:
            self.latent.kolmogorov_notify(self.t - self.t_0, type='backward')
        k = 0
        while(self.t > self.T - T):
            dprint(self.t)
            if (self.prev_obs > -1):
                t_prev = self.observations[self.prev_obs][0]
                if (np.isclose(self.t, t_prev)):
                    self.t = t_prev
                    y = self.observations[self.prev_obs][1]
                    b_u = self.latent.kolmogorov_marginal(type='backward')*self.om.observation_likelihood(y)
                    b_u = np.copy(b_u/np.sum(b_u))
                    dprint(b_u)
                    self.latent.kolmogorov_update(self.om.observation_likelihood(y), type='backward')
                    self.prev_obs -= 1
                    # doesn't have to be implemented but allows efficient memory allocation
                    if self.prev_obs >= 0:
                        self.latent.kolmogorov_notify(self.t - self.observations[self.prev_obs][0], type='backward')
                    else:
                        self.latent.kolmogorov_notify(self.t - self.t_0, type='backward')
                elif (t_prev < self.t):
                    if (t_prev > self.log_grid[self.prev_log]):
                        self.latent.kolmogorov_backward(self.t - t_prev)
                        self.t = np.round(t_prev, decimals=5)
                    else:
                        if not np.isclose(self.t - self.log_grid[self.prev_log], 0.):
                            self.latent.kolmogorov_backward(self.t - self.log_grid[self.prev_log])
                        self.t = np.round(self.log_grid[self.prev_log], decimals=5)
                        self.prev_log -= 1
                else:
                    print("Error (backward_pass): non-'causal' event. Aborting.")
                    return False
            else:
                self.latent.kolmogorov_backward(self.t - self.log_grid[self.prev_log])
                self.t = np.round(self.log_grid[self.prev_log], decimals=5)
                self.prev_log -= 1
            bs['time'][k] = self.t
            bs['marginal'][k] = self.latent.kolmogorov_marginal(type='backward')
            if _posterior == 1:
                bs['full'][k] = np.copy(self.latent.k.b_t)
            elif _posterior == 2 or _posterior == 3:
                bs['current']['input'][k] = self.latent.k.phib_t
                bs['current']['output'][k] = self.latent.k.psib_t
            if k%20 == 0:
                print("- step = " + str(k))
            k += 1
        # finally, delete trailing illegal times bc. of overallocation
        # and since we did the backward pass, also reverse the ordering
        for n in range(len(bs['time']) - len(self.observations), len(bs['time'])):
            if (np.isclose(bs['time'][n], -1.)):
                bs['time'] = np.flip(bs['time'][:n], axis=0)
                bs['marginal'] = np.flip(bs['marginal'][:n], axis=0)
                if _posterior == 1:
                    bs['full'] = np.flip(bs['full'][:n], axis=0)
                elif _posterior == 2 or _posterior == 3:
                    bs['current']['input'] = np.flip(bs['current']['input'][:n], axis=0)
                    bs['current']['output'] = np.flip(bs['current']['output'][:n], axis=0)
                break
        return bs

    def discrete_forward_backward(self, I, p_0, p_T, setup_tuple):
        # calculate the forward-backward posterior latent state probabilities
        total_duration = I[1] - I[0]
        # initialize the latent chain for the forward pass
        self.latent.p_0 = p_0
        self.latent.ss = None
        self.latent._steady_state_calculate()
        self.latent.kolmogorov_reset()
        self.latent.kolmogorov_setup(*setup_tuple)
        # First, do the forward pass
        self.set_time(I[0], stick_to_grid=True)
        ps = self.forward_pass(T=total_duration, _posterior=1)
        # initialize the latent chain for the backward pass
        self.latent.p_0 = p_T
        self.latent.ss = None
        self.latent._steady_state_calculate()
        self.latent.kolmogorov_reset()
        self.latent.kolmogorov_setup(*setup_tuple)
        # next, do the backward pass
        self.set_time(I[1], stick_to_grid=True)
        bs = self.backward_pass(T=total_duration, _posterior=1)
        # check along both arrays and build the posterior
        post = dict(time=np.zeros(len(ps['time'])), marginal=np.zeros([len(ps['time']), self.latent.num_states]))
        for n in range(len(ps['time'])):
            if not np.isclose(ps['time'][n], bs['time'][n]):
                # this should not be. Let's check if there is an error
                print("Times of both arrays not aligned")
                print(str([ps['time'][n], bs['time'][n]]))
                exit(0)
            else:
                post['marginal'][n] = np.sum(ps['full'][n]*bs['full'][n], axis=1)
                post['marginal'][n] /= np.sum(post['marginal'][n])
                post['time'][n] = ps['time'][n]
        # Return the posterior array
        return post

    def continuous_forward_backward(self, I, p_0, p_T, setup_tuple):
        # calculate the forward-backward posterior latent state probabilities
        total_duration = I[1] - I[0]
        # Do the forward and backward passes and then obtain the lists of integral inhomogeneities
        # initialize the latent chain for the forward pass
        self.latent.p_0 = p_0
        self.latent.ss = None
        self.latent._steady_state_calculate()
        self.latent.kolmogorov_reset()
        self.latent.kolmogorov_setup(*setup_tuple)
        # First, do the forward pass
        self.set_time(I[0], stick_to_grid=True)
        print("Running forward pass:")
        ps = self.forward_pass(T=total_duration, _posterior=3)
        # initialize the latent chain for the backward pass
        self.latent.p_0 = p_T
        self.latent.ss = None
        self.latent._steady_state_calculate()
        self.latent.kolmogorov_reset()
        self.latent.kolmogorov_setup(*setup_tuple, preset_normalization=deepcopy(ps['normalizers'].tolist()), initial='backward')
        # next, do the backward pass
        self.set_time(I[1], stick_to_grid=True)
        print("Running backward pass:")
        bs = self.backward_pass(T=total_duration, _posterior=3)
        #print(self.latent.k.normalization_factors)
        # We need a function that returns all likelihood within two time-windows
        # So, first write times and likelihoods in two arrays
        obs_times = self.observations[:, 0]
        obs_lhs = np.empty([len(obs_times), len(p_0)])
        for n in range(len(obs_times)):
            obs_lhs[n, :] = self.om.observation_likelihood(self.observations[n, 1])
        # sample the survival functions
        def fL(t, s):
            v = self.latent.sjt_dists[s].fun(t)
            v[np.isnan(v)] = 0.
            return v
        L = [None for _ in range(len(p_0))]
        for s in range(len(p_0)):
            L[s] = (lambda t, s=s: fL(t, s))
        # define a likelihood-by-path function
        def lh(t_l, t_h):
            b = (obs_times > t_l) & (obs_times < t_h)
            return np.prod(obs_lhs[b], axis=0)
        # retrieve the currents and the times
        # second builds first
        phia = ps['current']['input']
        psia = ps['current']['output']
        psib = bs['current']['output']
        phib = bs['current']['input']
        times = ps['time']
        norms = np.flip(np.array(ps['normalizers']))
        bnorms = np.append(norms[1:], 1.)
        #print(norms)
        #print(bnorms)
        N = len(times)
        # build an array of indices increasing only when there is a discontinuity
        # This will be used the following. We have disontinuities at positions
        # d_0, d_1, ... between t_l and t_h. At these points, sample L twice
        # so that we can apply trapz given obs_times
        t_ix = np.empty_like(times, dtype=int)
        t_ix[0] = 0
        t_diff = np.round(np.diff(times), decimals=5)
        t_ix[1:] = np.cumsum(np.logical_not(t_diff.astype(bool)).astype(int))
        forward_normalizers = np.empty_like(times)
        backward_normalizers = np.empty_like(times)
        for n in range(N):
            forward_normalizers[n] = np.prod(norms[:(t_ix[n] + 1)])
            backward_normalizers[n] = np.prod(bnorms[t_ix[n]:])
        #print(t_ix)
        #print(forward_normalizers)
        #print(backward_normalizers)
        # create a likelihood matrix
        Lh = np.zeros([N, N, self.latent.num_states])
        for n in range(N):
            for k in range(n, N):
                t_l = times[n] + 0.00001
                t_h = times[k] - 0.00001
                if (k > n) and (times[k - 1] == times[k]):
                    t_h = t_h + 0.00002
                if (n < N - 1) and (times[n + 1] == times[n]):
                    t_l = t_l - 0.00002
                Lh[n, k, :] = lh(t_l, t_h)
        # check along both arrays and build the posterior
        print("Solving posterior:")
        post = dict(time=np.zeros(len(ps['time'])), marginal=np.zeros([len(ps['time']), self.latent.num_states]))
        for n in range(len(ps['time'])):
            if not np.isclose(ps['time'][n], bs['time'][n]):
                # this should not be. Let's check if there is an error
                print("Times of both arrays not aligned")
                print(str([ps['time'][n], bs['time'][n]]))
                exit(0)
            elif n > 0 and n < len(ps['time']) - 1:
                print("- step = " + str(n))
                post['time'][n] = ps['time'][n]
                for s in range(self.latent.num_states):
                    ppsi = np.empty(n + 1)
                    for k in range(n + 1):
                        ppsi[k] = forward_normalizers[k]*phia[k, s]*np.trapz(L[s](times[n:] - times[k])*Lh[k, n:, s]*psib[n:, s]*backward_normalizers[n:], x=times[n:])
                        #ppsi[k] = phia[k, s]*np.trapz(L[s](times[n:] - times[k])*Lh[k, n:, s]*psib[n:, s], x=times[n:])
                    post['marginal'][n, s] = np.trapz(ppsi, times[:(n + 1)])
                #print('Marginal: ' + str(post['marginal'][n, :]) + ', Sum: ' + str(np.sum(post['marginal'][n, :])))
                post['marginal'][n, :] = post['marginal'][n, :]/np.sum(post['marginal'][n, :])
        #print(forward_normalizers)
        #print(backward_normalizers)
        post['time'][0] = ps['time'][0]
        post['marginal'][0, :] = p_0
        post['time'][-1] = ps['time'][-1]
        post['marginal'][-1, :] = p_T
        # plot the currents
        dtimes = []
        for n in range(len(times) - 1):
            if np.isclose(times[n + 1], times[n]):
                dtimes += [times[n]]
        #print(dtimes)
        def ppcallback(axs):
            for ax in axs:
                for d in dtimes:
                    ax.axvline(d, linestyle=':', color='#C0C0C0')
        # aux.plot_beautiful_sub_nxm( \
        #     [times, times], \
        #     [phia, psib], \
        #     [r'forward posterior current $\phi_{\alpha}$', r'backward posterior current $\psi_{\beta}$'], \
        #     [['', r'$\phi_{\alpha}(x,\,t)$'], [r'$t$', r'$\psi_{\beta}(x,\,t)$']], \
        #     partition='vertical', \
        #     suptitle=r'Normalized Forward and Backward Currents over $[0,\,T]$', \
        #     show=True, \
        #     savepath='./test_normalized_currents_example.pdf', \
        #     postprocess_callback=ppcallback)
        currents = dict()
        currents['normalizers'] = np.array(norms[1:])
        currents['forward'] = dict()
        currents['forward']['input'] = phia
        currents['forward']['output'] = psia
        currents['backward'] = dict()
        currents['backward']['output'] = psib
        currents['backward']['input'] = phib
        currents['time'] = times
        # Return the posterior array
        return (post, currents)
