#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:50:57 2020

@author: nicolai
"""

import numpy as np
from scipy import stats
from scipy import special
from scipy import optimize
from ..continuous_parametric import continuous_parametric as Continuous_Parametric_Distribution
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from random import randrange
import math

#def global_to_local_id(node):
#    # converts a global id to a local one
#    l_own_id = [node.par_ids + [node.own_id]].index(node.own_id)
#    return l_own_id
#
#def global_to_local_state(g_s, node):
#    # converts a global state list into a local state
#    # copy all substates, that are relevant
#    # returns a list with inherited order containing only the ancestral part of the Markov blanket
#    l_own_id = [node.par_ids + [node.own_id]].index(node.own_id)
#    l_idx = [g_s[i] for i in node.par_ids + [node.own_id]]
#    return l_idx, l_own_id
#
#def global_to_local_state_enum(g_s, node, smctbn):
#    # converts a global state list into a well enumerated local state
#    # copy all substates, that are relevant
#    # returns a list with inherited order containing only the ancestral part of the Markov blanket
#    l_idx = [g_s[i] for i in node.par_ids + [node.own_id]]
#    n_stat = [smctbn.nodes[i].num_states for i in node.par_ids + [node.own_id]]
#    pos = 1
#    l_s = 0
#    for i in range(0, len(n_stat)):
#        l_s += l_idx[i] * pos
#        pos *= n_stat[i]
#    return l_s
#
#def num_local_states(smctbn, node):
#    n_stat = [smctbn.nodes[i].num_states for i in node.par_ids + [node.own_id]]
#    return np.prod(n_stat)
#
#def local_state_enum(l_s, node, smctbn):
#    # transform a local state in a well enumerated local state
#    n_stat = [smctbn.nodes[i].num_states for i in node.par_ids + [node.own_id]]
#    pos = 1
#    l_e = 0
#    for i in range(0, len(n_stat)):
#        l_e += l_s[i] * pos
#        pos *= n_stat[i]
#    return l_e

# reparameterization functions
def rep_weibull(p):
    return [p[0], 1./np.power(p[1], p[0])]
def rep_gamma(p):
    return [p[0], 1./p[1]]
def rep_exp(p):
    return [1./p[0]]

def random_categorical(num_categories):
    # import numpy to use this
    d = [np.random.random_sample() for i in range(0, num_categories)]
    d = np.array(d) * (1./(np.sum(d)))
    return d.tolist()

def random_categorical_w_zero(num_categories_not_zero, pos):
    # import numpy to use this
    d = [np.random.random_sample() for i in range(0, num_categories_not_zero)]
    d = np.around(np.array(d) * (1./(np.sum(d))), 8)
    d = d.tolist()
    d = d[:pos] + [0] + d[pos:]
    return d

def staircase_categorical(num_other_states, pos, down_up):
    # import numpy to use this
    # down_up consists of a dictionary. The entries are as follows
    # "default" contains two lists, where the first goes down, the second goes up
    # each positive number containing the two lists will be assigned to that pos
    e = 'default'
    if pos in down_up.keys():
        e = pos
    N = num_other_states + 1
    d = [0 for _ in range(N)]
    dl = down_up[e][0]
    dr = down_up[e][1]
    for n in range(1, pos + 1):
        if n > len(dl):
            break
        d[pos - n] = down_up[e][0][-n]
    for n in range(N - pos):
        if n >= len(dr):
            break
        d[pos + n + 1] = down_up[e][1][n]
    return d

def random_gamma_distribution():
    # import scipy.stats and numpy to use this
    # draw both shape and rate from again a gamma distribution
    # parameters for the shape prior
    k_k = 20.
    k_r = 20.
    # parameters for the rate prior
    l_k = 1.5
    l_r = 0.5
    # draw parameters, create the distribution object and hand it over
    k = stats.gamma.rvs(k_k, size=1, scale=(1./l_k))[0]
    #print(stats.gamma.stats(k_k, scale=(1./l_k), moments='mv'))
    r = stats.gamma.rvs(k_r, size=1, scale=(1./l_r))[0]
    #print(stats.gamma.stats(k_r, scale=(1./l_r), moments='mv'))
    dist = Continuous_Parametric_Distribution(stats.gamma, name='gamma', loc=None, scale=(1/r), a=k, reparam=rep_gamma)
    return dist

def fixed_shape_random_rate_gamma_distribution(shape):
    # import scipy.stats and numpy to use this
    # draw both shape and rate from again a gamma distribution
    # parameters for the shape prior
    k_r = shape*5.
    # parameters for the rate prior
    l_r = 10.
    # draw parameters, create the distribution object and hand it over
    k = shape
    r = stats.gamma.rvs(k_r, size=1, scale=(1./l_r))[0]
    dist = Continuous_Parametric_Distribution(stats.gamma, name='gamma', loc=None, scale=(1./r), a=k, reparam=rep_gamma)
    return dist

def random_exponential_distribution():
    # import scipy.stats and numpy to use this
    # draw both shape and rate from again a gamma distribution
    # parameters for the shape prior
    k_r = 5.
    # parameters for the rate prior
    l_r = 3.
    # draw parameters, create the distribution object and hand it over
    r = stats.gamma.rvs(k_r, size=1, scale=(1./l_r))[0]
    dist = Continuous_Parametric_Distribution(stats.expon, name='exponential', loc=None, scale=(1./r), reparam=rep_exp)
    return dist

def exponential_distribution(r):
    dist = Continuous_Parametric_Distribution(stats.expon, name='exponential', loc=None, scale=(1./r), reparam=rep_exp)
    return dist

def fixed_shape_random_rate_weibull_distribution(shape):
    # import scipy.stats and numpy to use this
    # draw both shape and rate from again a gamma distribution
    wanted = (1.08 + 0.04*(1 + 1./shape))/(special.gamma(1 + 1./shape))  # we want a mean of 1 for the drawn sojourn times
    # manual correction for the mean to achieve roughly the same amount of average transitions for different shapes
    c_v = np.power(2./(3.*3.), 0.5)  # constant coefficient of variation
    var_0 = 3./(3.*3.)
    # draw parameters, create the distribution object and hand it over
    meen = (lambda r: (1./np.power(r, 1./shape)) - wanted)
    meen_prime = (lambda r: (-1./(shape*np.power(r, 1 + 1./shape))))
    meen_prime2 = (lambda r: (1./shape + 1./np.power(shape, 2))*(1./np.power(r, 2 + 1./shape)))
    r_mean = optimize.newton(meen, 0.1, fprime=meen_prime, fprime2=meen_prime2)
    # parameters for the rate prior
    l_r = 1./(np.power(c_v, 2) * r_mean)
    #l_r = r_mean/var_0
    # parameters for the shape prior
    k_r = r_mean*l_r
    k = shape
    r = stats.gamma.rvs(k_r, size=1, scale=(1./l_r))[0]
    #print(stats.gamma.stats(k_r, scale=(1./l_r), moments='mv'))
    #print(str((1./np.power(r_mean, 1./shape))*(special.gamma(1 + 1./shape))))
    dist = Continuous_Parametric_Distribution(stats.weibull_min, name='weibull', loc=None, scale=(1./np.power(r, 1./k)), a=k, reparam=rep_weibull)
    return dist

def random_weibull_distribution():
    # import scipy.stats and numpy to use this
    # draw both shape and rate from again a gamma distribution
    # parameters for the shape prior
    k_k = 40.
    k_r = 50.
    # parameters for the rate prior
    l_k = 7.792
    l_r = 7.9231
    # draw parameters, create the distribution object and hand it over
    k = stats.gamma.rvs(k_k, size=1, scale=(1./l_k))[0]
    #print(stats.gamma.stats(k_k, scale=(1./l_k), moments='mv'))
    r = stats.gamma.rvs(k_r, size=1, scale=(1./l_r))[0]
    #print(stats.gamma.stats(k_r, scale=(1./l_r), moments='mv'))
    dist = Continuous_Parametric_Distribution(stats.weibull_min, name='weibull', loc=None, scale=(1./np.power(r, 1./k)), a=k, reparam=rep_weibull)
    return dist

def random_graph(num_nodes, range_pars):
    # draw a random graph
    A = []
    for i in range(0, num_nodes):
        d = random_categorical(num_nodes - 1)
        n = random_categorical(range_pars[1] - range_pars[0] + 1)
        # this gives a random number of children
        n = sample_categorical(n) + range_pars[0]
        # now, we draw the indices of the n max values from d
        idx = np.argpartition(d, -n)[-n:]
        edges = [0] * num_nodes
        for j in idx:
            if j >= i:
                edges[j + 1] = 1
            else:
                edges[j] = 1
        A += [edges]
    return np.transpose(A).tolist()

def plot_graph(A, labels):
    G = nx.DiGraph()
    G.add_nodes_from(labels)
    for i in range(len(A)):
     for j in range(len(A)):
       if A[i][j] == 1:
          G.add_edge(labels[i], labels[j])
    nx.draw(G, with_labels=True)
    plt.show()

def uniform_range(r):
    return (r[1] - r[0])*np.random.random_sample() + r[0]

def uniform_range_int(r):
    return randrange(r[1] - r[0] + 1) + r[0]

def sample_categorical(pmf):
    # import numpy to use this
    probs = deepcopy(pmf)
    smpl = np.random.random_sample()
    sm = 0.
    c = -1
    while sm < smpl:
        sm += probs.pop(0)
        c += 1
    return c

def pad_zeros(a, n):
    m = len(a)
    o = np.zeros(m*(n + 1))
    o[::(n + 1)] = a
    return o

def vector_range(u, v):
    # gives all combinations of numbers from u to v in range of every single dimension
    lst = []
    for i in range(u[0], v[0]):
        lst += vector_range_helper([i], u[1:], v[1:])
    return lst

def vector_range_helper(lst, u, v):
    # gives all combinations of numbers from u to v in range of every single dimension
    # helper function to handover the list prefix
    if u == []:
        return [deepcopy(lst)]
    else:
        ret = []
        for i in range(u[0], v[0]):
            ret += vector_range_helper(lst + [i], u[1:], v[1:])
    return ret

def split_trajectory(tr, every):
    # split trajectory into chunks with history object
    history = None
    split = [None] * int(len(tr)/every)
    for n in range(len(split)):
        split[n] = [None] * 2
        split[n][0] = [tr[n*every]]
        split[n][1] = deepcopy(history)
        history = np.zeros(len(tr[0]['states']))
        for i in range(every - int(n == len(split) - 1)):
            dt = tr[n*every + i + 1]['timestamp'] - tr[n*every + i]['timestamp']
            df = np.array(tr[n*every + i + 1]['states']) - np.array(tr[n*every + i]['states'])
            history += dt
            split[n][0] += [tr[n*every + i + 1]]
            history[int(np.nonzero(df)[0])] = 0
    return split

def categorials_to_mean_and_var(ps, levels):
    ys = np.zeros([len(ps), 3])
    for n in range(len(ps)):
        ys[n][1] = np.sum(ps[n]*levels)
        var = np.sum(ps[n]*levels*levels) - ys[n][1]*ys[n][1]
        ys[n][0] = ys[n][1] - var
        ys[n][2] = ys[n][1] + var
    return ys

def categorials_to_quantiles(ps, quantiles, levels):
    ys = np.zeros([len(ps), len(quantiles)])
    cdfs = np.cumsum(ps, axis=1)
    cdfs = cdfs/cdfs[:, -1, np.newaxis]
    for n in range(len(ps)):
        for m in range(len(quantiles)):
            ys[n, m] = levels[next((k for k in range(len(levels)) if cdfs[n, k] > quantiles[m]))]
    return ys

def categorials_to_mode_and_quantiles(ps, quantiles, levels):
    ys = np.zeros([len(ps), len(quantiles) + 1])
    cdfs = np.cumsum(ps, axis=1)
    cdfs = cdfs/cdfs[:, -1, np.newaxis]
    for n in range(len(ps)):
        for m in range(len(quantiles)):
            ys[n, m + 1] = levels[next((k for k in range(len(levels)) if cdfs[n, k] > quantiles[m]))]
        ys[n, 0] = levels[np.argmax(ps[n])]
    return ys

# use raw strings. We use latex
def plot_beautiful_sub_nxm(xs, ys, titles, axis_labels, partition='vertical', suptitle=None, savepath=None, autoshare=False, show=False, postprocess_callback=None):
    #marker = itertools.cycle(('o', 'v', '^'))
    #lines  = itertools.cycle(('-', '-.', ':'))
    #colors  = itertools.cycle(('#AA4371', '#C4961A', '#00AFBB'))
    plt.rcParams['text.usetex'] = True
    N = len(xs)
    if (len(ys) != len(xs) or len(titles) != len(xs)):
        raise ValueError
    fig = None
    axs = None
    if partition == 'vertical':
        fig, axs = plt.subplots(N, 1)
    elif partition == 'horizontal':
        fig, axs = plt.subplots(1, N)
    else:
        raise ValueError
    if suptitle is not None:
        fig.suptitle(suptitle)
    for n in range(N):
        axs[n].plot(xs[n], ys[n])
        axs[n].title.set_text(titles[n])
        axs[n].set_xlabel(axis_labels[n][0])
        axs[n].set_ylabel(axis_labels[n][1])
        axs[n].set_xlim([xs[n][0], xs[n][-1]])
    # do special temporary modifications here
    if postprocess_callback is not None:
        postprocess_callback(axs)
    # finally save and plot
    fig.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)
    if show:
        fig.show(warn=False)
        plt.show()
