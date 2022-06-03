# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 12:28:33 2020

@author: user
"""

import scipy.linalg as sla
import scipy.stats as stats
import numpy.linalg as nla
import numpy as np
import matplotlib.pyplot as plt

class D1D_C1D_White_Observation_Model:
    # the constructor of the SMCTBN expects a set of SMCTBNNodes, an Adjacency matrix, a randomizer dictionary
    def __init__(self, means, std):
        self.means = means
        self.latent_size = len(means)
        self.std = std
        self.noise = [None for _ in range(self.latent_size)]
        for s in range(self.latent_size):
            self.noise[s] = stats.norm(loc=self.means[s], scale=std)

    def observation_likelihood(self, y):
        lhs = np.zeros(self.latent_size)
        for s in range(self.latent_size):
            lhs[s] = self.noise[s].pdf(y)
        return lhs

    def draw_observation(self, latent_state):
        return self.noise[latent_state].rvs()

    def oberservation_model_plot(self, ax, grayscale=False):
        rang = [np.min(self.means) - 4*self.std, np.max(self.means) + 4*self.std]
        dx = 0.01
        xs = np.arange(rang[0], rang[1] + dx/2., dx)
        ys = np.zeros([self.latent_size, len(xs)])
        for s in range(self.latent_size):
            ys[s] = self.noise[s].pdf(xs)
        for s in range(self.latent_size):
            if grayscale:
                ax.hlines(self.means[s], 0, self.noise[s].pdf(self.means[s]) + self.std, color='#a0a0a0', linestyles='dotted')  # Stems
            ax.hlines(self.means[s], 0, self.noise[s].pdf(self.means[s]) + self.std, 'C' + str(s), linestyles='dotted')  # Stems
            #ax.plot(self.noise[s].pdf(self.means[s]) + self.std, self.means[s], '>')  # Stem ends
            #ax.stem([self.means[s]], [self.noise[s].pdf(self.means[s]) + self.std], linefmt='C' + str(s) + '--', markerfmt='C' + str(s) + '^', use_line_collection=True)
        ax.set_prop_cycle(None)
        if grayscale:
            ax.plot(np.transpose(ys), xs, color='#a0a0a0')
        ax.plot(np.transpose(ys), xs)
        ax.plot(np.zeros(len(xs)), xs, color='#a0a0a0')
