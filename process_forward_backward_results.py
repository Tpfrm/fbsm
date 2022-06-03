# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:59:05 2020

@author: user
"""

import lib.aux.auxilliary as aux
from lib.discretized_ctsmc import Discretized_CTSMC as CTSMC
from lib.integro_differential_auxilliary import discretizedCTSMC_to_integrodifferentialCTSMC
from lib.integrodifferential_ctsmc import integrodifferential_CTSMC
from lib.integral_ctsmc import integral_CTSMC
import scipy.linalg as sla
import numpy.linalg as nla
import numpy as np
import matplotlib.pyplot as plt
import dill as pickle
import time
from copy import deepcopy
from lib.d1d_c1d_white_observation_model import D1D_C1D_White_Observation_Model as Om
from lib.hmm import HMM
import re
import configparser
import os

# def plot_clock(ctsmc):
#     plt.figure()
#     plt.plot(ctsmc.k.p_t.T[:250])
#     plt.show()

if __name__ == '__main__':

    # read in trajectories and observations
    working_dir = './results_raw/forward_backward'
    file_name = 'id1654195471_posterior_6s17o_evaluation_discrete.pkl'
    print("Handover file: " + file_name)
    content = None
    print("Reading in simulation data... ", end="")
    with open(working_dir + "/" + file_name, "rb") as fyle:
        content = pickle.load(fyle)
    print("done.")

    # process and sort out some parameters
    machine = content['machine']
    config = content['config']
    hyper = content['hyper']
    ps = content['post']
    ctsmc = content['model']['chain']
    om = content['model']['observation']
    tr = content['source']['trajectory']
    ob = content['source']['observations']
    id_ = re.search("id(.+?)_.*", file_name, re.IGNORECASE).groups(1)[0]
    sjt_type = re.search(".*[0-9]o_(.+?)_.*", file_name, re.IGNORECASE).groups(1)[0]

    # important time-related quantities
    dt = 0.01#ps['time'][1] - ps['time'][0]
    T = float(tr[0][-1]['time'])

    # plot the stuff
    ctr = tr[0]
    t = np.arange(0., T, dt)
    s = np.zeros(len(t))
    n = 0
    for c in range(len(t)):
        if (ctr[n]['time'] <= t[c]):
            n += 1
        s[c] = ctr[n - 1]['state']
    lv = np.ones_like(t)
    obs = np.transpose(ob)
    ax1width = T
    ax2width = om.noise[0].pdf(om.means[0]) + om.std
    axsheight = (np.max(om.means) + 4*om.std - (np.min(om.means) - 4*om.std))
    fig, (ax1) = plt.subplots(1, 1, gridspec_kw={'width_ratios': [ax1width]}, figsize=(0.6*(ax1width), 0.6*axsheight))
    ax1.plot(t, np.outer(lv, om.means), ':', color='#a0a0a0')
    ax1.plot(t, s + 1, color='#000000')

    ys = np.transpose(aux.categorials_to_mean_and_var(ps['marginal'], om.means))
    #ys = np.transpose(aux.categorials_to_quantiles(ps['marginal'], [0.01, 0.5, 0.99], om.means))
    discont = [0]
    for n in range(1, len(ps['time'])):
        if (np.isclose(ps['time'][n], ps['time'][n - 1])):
            discont += [n]
    discont += [len(ps['time'])]
    for n in range(len(discont) - 1):
        ax1.plot(ps['time'][discont[n]:(discont[n + 1])], ys[1][discont[n]:(discont[n + 1])], 'C0')
        ax1.plot(ps['time'][discont[n]:(discont[n + 1])], ys[0][discont[n]:(discont[n + 1])], 'C0--', alpha=0.1)
        ax1.plot(ps['time'][discont[n]:(discont[n + 1])], ys[2][discont[n]:(discont[n + 1])], 'C0--', alpha=0.1)
        ax1.fill_between(ps['time'][discont[n]:(discont[n + 1])], ys[0][discont[n]:(discont[n + 1])], ys[2][discont[n]:(discont[n + 1])], color='C0', alpha=0.1)
        ax1.set_prop_cycle(None)
    ax1.plot(obs[0], obs[1], 'x', markeredgewidth=1, markersize=12, color='#000000')

    #ax1.set_ylim([np.min(om.means) - 4*om.std, np.max(om.means) + 4*om.std])
    ax1.set_ylim([np.min(om.means) - 1, np.max(om.means) + 1])
    ax1.set_xlim([0, ax1width])
    #ax1.set_aspect(0.7, adjustable='box')

    print(discont)
    for n in range(len(ps['time'])):
        print('Time: ' + str(ps['time'][n]) + ', Mean: ' + str(ys[1][n]))
    #om.oberservation_model_plot(ax2)
    #ax2.set_ylim([np.min(om.means) - 4*om.std, np.max(om.means) + 4*om.std])
    #ax2.set_ylim([np.min(om.means) - 1, np.max(om.means) + 1])
    #ax2.set_xlim([0, ax2width])
    #ax2.yaxis.tick_right()
    #ax2.set_aspect(1.3, adjustable='box')
    fig.tight_layout()
    #fig.show(warn=False)
    #plt.show()

    # save the results
    results_dir = 'results/forward_backward'
    print("Writing out human readable results to " + results_dir + "... ", end="")
    experiment_dir = os.path.join(results_dir, 'id' + id_)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # first save the plots
    plt.savefig(os.path.join(experiment_dir, 'plots.png'))

    # then, save the stats and description of the experiment
    stats_dict = configparser.ConfigParser()
    stats_dict['Execution'] = dict()
    stats_dict['Execution']['Experiment ID'] = id_
    stats_dict['Execution']['Algorithm'] = 'Marginal latent state estimation (' + str(config['action']) + ')'
    if config['method'] == 'discrete':
        stats_dict['Execution']['Configuration'] = 'discrete-time approximation'
    elif config['method'] == 'continuous':
        if config['action'] == 'posterior' or config['equation'] == 'integral':
            stats_dict['Execution']['Configuration'] = 'continuous-time integration via currents'
        elif config['equation'] == 'integrodifferential':
            stats_dict['Execution']['Configuration'] = 'continuous-time integration via integrodifferential evolution equations'
    stats_dict['Execution']['Python version'] = str(machine['python_version']).replace('\n', ' ')
    #stats_dict['Execution']['Imported python modules'] = str(machine['loaded_modules'])  # don't print this by default. It is too long
    stats_dict['Execution']['Runtime'] = str(round(machine['runtime'], 3)) + ' sec.'
    stats_dict['Machine Info'] = dict()
    stats_dict['Machine Info']['Processor'] = str(machine['processor'])
    stats_dict['Machine Info']['RAM'] = str(machine['ram'])
    stats_dict['Machine Info']['CPUs'] = 'physical (' + str(machine['cpu_physical']) + '), logical (' + str(machine['cpu_logical']) + '), affinity (' + str(machine['cpu_affinity']) + ')'
    stats_dict['Machine Info']['OS'] = str(machine['os'])
    stats_dict['Data'] = dict()
    stats_dict['Data']['Trajectory length'] = str(tr[0][-1]['time'] - tr[0][0]['time'])
    stats_dict['Data']['Number of observations'] = str(len(ob))
    stats_dict['Data']['Boundary'] = 'steady state'
    stats_dict['Chain Model'] = dict()
    stats_dict['Chain Model']['Implementation'] = str(ctsmc.name)
    stats_dict['Chain Model']['Number of states'] = str(ctsmc.num_states)
    stats_dict['Chain Model']['Embedded MC transition probabilities'] = '\n' + str('\n'.join([str(lst) for lst in ctsmc.jp_dists.tolist()]))
    stats_dict['Chain Model']['Waiting time distributions'] = '\n' + str('\n'.join(['(type = ' + str(ctsmc.sjt_dists[s].name) + ', shape = ' + str(ctsmc.sjt_dists[s].a) + ', scale = ' + str(ctsmc.sjt_dists[s].scale) + ')' for s in range(ctsmc.num_states)]))
    stats_dict['Observation Model'] = dict()
    stats_dict['Observation Model']['Values'] = str(om.means)
    stats_dict['Observation Model']['Noise Model'] = 'additive Gaussian noise'  # no alternative to that right now
    stats_dict['Observation Model']['Standard deviation'] = str(om.std)
    stats_dict['Hyperparametrization'] = dict()
    stats_dict['Hyperparametrization']['Observation process'] = '(type = ' + str(hyper['observation_distribution'].name) + ', shape = ' + str(hyper['observation_distribution'].a) + ', scale = ' + str(hyper['observation_distribution'].scale) + ')'
    stats_dict['Hyperparametrization']['Parametrization setting'] = hyper['parametrization_type']

    with open(os.path.join(experiment_dir, 'info.txt'), 'w') as fyle:
        stats_dict.write(fyle)
    print("done.")
