# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 23:59:05 2020

@author: user
"""

import lib.aux.auxilliary as aux
from lib.discretized_ctsmc import Discretized_CTSMC as CTSMC
from lib.integro_differential_auxilliary import discretizedCTSMC_to_integrodifferentialCTSMC
from lib.integro_differential_auxilliary import discretizedCTSMC_to_integralCTSMC
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
import timeit
import sys
import platform
import psutil

# def plot_clock(ctsmc):
#     plt.figure()
#     plt.plot(ctsmc.k.p_t.T[:250])
#     plt.show()

if __name__ == '__main__':

    # config
    action = 'posterior'
    method = 'discrete'
    equation = 'integral'
    grid = 'uniform'

    # read in trajectories and observations
    working_dir = './evidence'
    file_name = 'id1654195471_evidence_6s_evaluation_04_len1_time13.pkl'
    print("Handover file: " + file_name)
    content = None
    print("Reading in simulation data... ", end="")
    with open(working_dir + "/" + file_name, "rb") as fyle:
        content = pickle.load(fyle)
    print("done.")

    # process and sort out some parameters
    dctsmc = content['ctsmc']
    hyper = content['hyper']
    tr = content['tr']
    ob = content['ob']
    om = content['om']
    id_ = re.search("id(.+?)_.*", file_name, re.IGNORECASE).groups(1)[0]
    sjt_type = re.search(".*[0-9]s_(.+?)_.*", file_name, re.IGNORECASE).groups(1)[0]

    # for this script, we only need one trajectory. Take the first one
    tr = tr[0]
    ob = ob[0]#[0:2]

    # add initial end terminal windows, so that we can infer the boundary
    # don't add the offset on the initial value and add double to the final
    # don't forget the observations
    offset = float(tr[0][-1]['time'])/2
    for n in range(len(tr[0])):
        if n > 0:
            tr[0][n]['time'] += offset
    tr[0][-1]['time'] += offset
    for n in range(len(ob)):
        ob[n][0] += offset

    dt = 0.001
    Dt = 100.*dt
    T = float(tr[0][-1]['time'])

    t = np.arange(0, T, dt)

    # decide if we want to use the discrete or continuous version
    if method == 'discrete':
        ctsmc = CTSMC(dctsmc.p_0, randomizer=None)
        for s in range(0, ctsmc.num_states):
            ctsmc.set_sjt_dist(s, deepcopy(dctsmc.sjt_dists[s]))
            ctsmc.set_jp_dist(s, deepcopy(dctsmc.jp_dists[s]))
        ctsmc._steady_state_calculate()
    elif method == 'continuous':
        if equation == 'integral' or action == 'posterior':
            ctsmc = discretizedCTSMC_to_integralCTSMC(dctsmc)
        elif equation == 'integrodifferential':
            ctsmc = discretizedCTSMC_to_integrodifferentialCTSMC(dctsmc)
    p_0 = np.copy(ctsmc.p_0)

    # get ready for filtering
    if action == 'forward':
        ctsmc.kolmogorov_setup(1*T, dt)
    elif action == 'backward':
        ctsmc.p_0 = np.ones(dctsmc.num_states)/dctsmc.num_states
        if method == 'discrete':
            ctsmc.kolmogorov_setup(1*T, dt)
        elif method == 'continuous':
            ctsmc.kolmogorov_setup(1*T, dt, initial='backward')

    # initialize the hmm object
    hmm = HMM(ctsmc, om, 0, T, Dt)
    hmm.observe(ob)

    ps = None
    runtime = timeit.default_timer()
    if action == 'forward':
        hmm.set_time(0, stick_to_grid=True)
        if grid == 'adaptive':
            ps = hmm.forward_pass(T, grid=True)
        else:
            ps = hmm.forward_pass(T)
    elif action == 'backward':
        hmm.set_time(T, stick_to_grid=True)
        ps = hmm.backward_pass(T)
    elif action == 'posterior':
        if method == 'discrete':
            ps = hmm.discrete_forward_backward([0, T], p_0, np.ones(dctsmc.num_states)/dctsmc.num_states, (1*T, dt))
        else:
            ps, currents = hmm.continuous_forward_backward([0, T], p_0, np.ones(dctsmc.num_states)/dctsmc.num_states, (1*T, dt))
    runtime = timeit.default_timer() - runtime

    # remove initial and terminal windows
    for n in range(len(ob)):
        ob[n][0] -= offset
    for n in range(len(tr[0])):
        if n > 0:
            tr[0][n]['time'] -= offset
    tr[0][-1]['time'] -= offset
    ps_ix = np.logical_and(ps['time'] >= offset, ps['time'] <= ps['time'][-1] - offset)
    ps_clean = dict()
    ps_clean['time'] = ps['time'][ps_ix]
    ps_clean['marginal'] = ps['marginal'][ps_ix, :]
    ps_clean['time'] -= offset
    ps_clean['grid'] = None
    if grid == 'adaptive':
        total_len = 0
        for l in ps['grid']:
            total_len += len(l)
        ps_clean['grid'] = np.zeros(total_len)
        current_len = 0
        for l in ps['grid']:
            ps_clean['grid'][current_len:(current_len + len(l))] = l - offset
            current_len += len(l)
    print(ps_clean['grid'])

    # save config
    config = dict()
    config['action'] = action
    config['method'] = method
    config['equation'] = equation
    config['grid'] = grid

    # get machine info
    machine = dict()
    machine['runtime'] = runtime
    machine['python_version'] = sys.version
    machine['loaded_modules'] = list(sys.modules.keys())
    machine['processor'] = platform.processor()
    machine['ram'] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GiB"
    machine['cpu_physical'] = psutil.cpu_count(logical=False)
    machine['cpu_logical'] = psutil.cpu_count(logical=True)
    machine['cpu_affinity'] = len(psutil.Process().cpu_affinity())
    machine['os'] = platform.system()

    # write out posterior data
    working_dir = './results_raw/forward_backward'
    file_name = 'id' + str(id_) + '_' + action + '_' + str(ctsmc.num_states) + 's' + str(len(ob)) + 'o_' + str(sjt_type) + '_' + method + '.pkl'
    if action == 'posterior' and method == 'continuous':
        content = {'config': config, 'hyper': hyper, 'machine': machine, 'post': ps_clean, 'currents': currents, 'source': {'trajectory': tr, 'observations': ob}, 'model': {'chain': ctsmc, 'observation': om}}
    else:
        content = {'config': config, 'hyper': hyper, 'machine': machine, 'post': ps_clean, 'source': {'trajectory': tr, 'observations': ob}, 'model': {'chain': ctsmc, 'observation': om}}
    print("Writing out simulation data to " + file_name + "... ", end="")
    with open(working_dir + "/" + file_name, "wb") as fyle:
        pickle.dump(content, fyle)
    print("done.")
