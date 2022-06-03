# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:59:05 2020

@author: user
"""

from lib.aux import auxilliary as aux
from lib.discretized_ctsmc import Discretized_CTSMC as CTSMC
import scipy.linalg as sla
import numpy.linalg as nla
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import time
from copy import deepcopy
from lib.d1d_c1d_white_observation_model import D1D_C1D_White_Observation_Model as Om
from lib.continuous_parametric import continuous_parametric
import math
import scipy.stats as stats

def plot_clock(ctsmc):
    plt.figure()
    plt.plot(ctsmc.k.p_t.T[:250])
    plt.show()

if __name__ == '__main__':

    # Configuration
    num_states = 3
    simulation_end = 10.
    simulation_endtype = 'time'  # allowed 'time' or 'jumps'
    simulation_trajectories = 1
    sjt_parameterization = 'random_weibull'
    fixed_shape = 5*np.pi
    working_dir = './evidence'

    # observation point process
    obs_loc = 0.
    obs_a = 2.
    obs_scale = 1./4.
    obs_noise_std = 0.5
    obs_distribution = continuous_parametric(stats.gamma, name='gamma point process', scale=obs_scale, a=obs_a)

    # Define the randomizer dictionary
    if (sjt_parameterization == 'random_exp'):
        randomizer = dict(sjt=aux.random_exponential_distribution, jp=aux.random_categorical_w_zero)
    elif (sjt_parameterization == 'random_gamma'):
        randomizer = dict(sjt=aux.random_gamma_distribution, jp=aux.random_categorical_w_zero)
    elif (sjt_parameterization == 'gamma_fixed_shape'):
        sjt_generator = lambda: aux.fixed_shape_random_rate_gamma_distribution(fixed_shape)
        randomizer = dict(sjt=sjt_generator, jp=aux.random_categorical_w_zero)
    elif (sjt_parameterization == 'weibull_fixed_shape'):
        sjt_generator = lambda: aux.fixed_shape_random_rate_weibull_distribution(fixed_shape)
        randomizer = dict(sjt=sjt_generator, jp=aux.random_categorical_w_zero)
    elif (sjt_parameterization == 'random_weibull'):
        randomizer = dict(sjt=aux.random_weibull_distribution, jp=aux.random_categorical_w_zero)
    elif (sjt_parameterization == 'evaluation_02'):
        sjt_generator = lambda: aux.fixed_shape_random_rate_gamma_distribution(fixed_shape)
        down_up = dict({'default': [[0.],[1.]], (num_states - 1): [[1.] + [0. for _ in range(num_states - 2)],[]]})
        randomizer = dict(sjt=sjt_generator, jp=(lambda n, p, d=down_up: aux.staircase_categorical(n, p, d)))
    elif (sjt_parameterization == 'evaluation_03'):
        sjt_generator = lambda: aux.fixed_shape_random_rate_weibull_distribution(fixed_shape)
        randomizer = dict(sjt=sjt_generator, jp=aux.random_categorical_w_zero)
    elif (sjt_parameterization == 'evaluation_04'):
        sjt_generator = lambda: aux.fixed_shape_random_rate_gamma_distribution(fixed_shape)
        randomizer = dict(sjt=sjt_generator, jp=aux.random_categorical_w_zero)

    # create the CTSMC. After the first setup, take the steady state as initial condition
    ctsmc = CTSMC(np.empty(num_states), randomizer)
    ctsmc.p_0 = np.zeros_like(ctsmc.ss.p_inf)
    ctsmc.p_0[0] = 1

    dt = 0.01
    T = simulation_end

    if T > simulation_end:
        raise ValueError('Either simulation end time or sampling times< are wrong.')

    t = np.arange(0, T, dt)
    # print the sojourn time distributions
    pdf = np.zeros([ctsmc.num_states, len(t)])
    lf = np.zeros([ctsmc.num_states, len(t)])
    for s in range(ctsmc.num_states):
        pdf[s] = ctsmc.sjt_dists[s].fun(t)
        lf[s] = ctsmc.lf[s](t)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(t, np.transpose(pdf))
    plt.subplot(1, 2, 2)
    plt.plot(t, np.transpose(lf))
    plt.tight_layout()
    plt.show()

    print("")
    ctsmc.steady_state_print()
    print("")

    om = Om(list(range(1, num_states + 1)), obs_noise_std)
    #om.oberservation_model_plot()

    # simulate 'simulation_trajectories' trajectories
    tr = []
    print("Sampling trajectories (" + str(simulation_trajectories) + " total)...")
    for i in range(simulation_trajectories):
        print(str(i) + ", ", end="")
        history = np.random.uniform(0, 1)
        tr += [[ctsmc.sim_run(simulation_end, endby=simulation_endtype, history=history), deepcopy(history)]]
    print("done")

    # draw random samples using the point process defined and obstruct them using the observation model
    od = obs_distribution.dist(a=obs_a, loc=obs_loc, scale=obs_scale)
    ob = []
    print("Sampling observations (" + str(simulation_trajectories) + " total)...")
    for i in range(simulation_trajectories):
        print(str(i) + ", ", end="")
        obs = [None, None]
        obs[0] = [np.round(od.rvs(), decimals=2)]
        ctr = tr[i][0]
        while (obs[0][-1] < ctr[-1]['time']):
            obs[0] += [np.round(obs[0][-1] + od.rvs(), decimals=2)]
        obs[0] = obs[0][:-1]
        obs[1] = [0 for _ in range(len(obs[0]))]
        c = 0
        n = 0
        while c < len(obs[0]):
            if (ctr[n]['time'] > obs[0][c]):
                obs[1][c] = om.draw_observation(ctr[n - 1]['state'])
                c += 1
            else:
                n += 1
        ob += [np.transpose(obs).tolist()]
    print("done")

    # plot the last trajectory
    ctr = tr[-1][0]
    t = np.arange(0., T, dt)
    s = np.zeros(len(t))
    n = 0
    for c in range(len(t)):
        if (ctr[n]['time'] <= t[c]):
            n += 1
        s[c] = ctr[n - 1]['state']
    lv = np.ones_like(t)
    ax1width = T
    ax2width = om.noise[0].pdf(om.means[0]) + om.std
    axsheight = np.max(om.means) + 4*om.std - (np.min(om.means) - 4*om.std)
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [ax1width, ax2width]}, figsize=(0.6*(ax1width + ax2width), 0.6*axsheight))
    ax1.plot(t, np.outer(lv, om.means), ':')
    ax1.plot(t, s + 1, color='#000000')
    ax1.plot(obs[0], obs[1], 'x', markeredgewidth=4, markersize=12, color='#ff0000')
    #ax1.set_ylim([np.min(om.means) - 4*om.std, np.max(om.means) + 4*om.std])
    ax1.set_ylim([np.min(om.means) - 1, np.max(om.means) + 1])
    ax1.set_xlim([0, ax1width])
    #ax1.set_aspect(0.7, adjustable='box')

    om.oberservation_model_plot(ax2)
    #ax2.set_ylim([np.min(om.means) - 4*om.std, np.max(om.means) + 4*om.std])
    ax2.set_ylim([np.min(om.means) - 1, np.max(om.means) + 1])
    ax2.set_xlim([0, ax2width])
    ax2.yaxis.tick_right()
    #ax2.set_aspect(1, adjustable='box')
    fig.tight_layout()
    fig.show(warn=False)

    # create hyper dict containing information about data and model generation
    hyper = dict()
    hyper['observation_distribution'] = obs_distribution
    hyper['parametrization_type'] = sjt_parameterization

    # prepare
    len_avg_tr = int(math.floor((1./len(tr))*np.sum([len(lst[0]) for lst in tr])))

    # write out the results and store persistently
    file_name = "id" + str(math.floor(time.time())) + "_evidence_" + str(num_states) + "s_" + sjt_parameterization + "_len" + str(len(tr)) + "_" + simulation_endtype + str(len_avg_tr)

    # gather anything useful
    pkl_data = dict(ctsmc=ctsmc, tr=tr, ob=ob, om=om, hyper=hyper)

    # save the stuff to unique file
    with open(working_dir + "/" + file_name + ".pkl", "wb") as fyle:
        pickle.dump(pkl_data, fyle)
