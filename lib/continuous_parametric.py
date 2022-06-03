#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 09:07:16 2020

@author: nicolai
"""

from .aux import auxilliary as auxo

class continuous_parametric:
    def __init__(self, scipy_stats_dist, name='unspecified', loc=None, scale=None, a=None, reparam=None):
        self.dist = scipy_stats_dist
        self.name = name
        if (loc is None):
            self.loc = 0
        else:
            self.loc = 0
        if (scale is None):
            self.scale = 1
        else:
            self.scale = scale
        self.a = a
        self.reparam = reparam

    def draw(self, size=1):
        if (self.a is not None):
            return self.dist.rvs(self.a, size=size, loc=self.loc, scale=self.scale)
        else:
            return self.dist.rvs(size=size, loc=self.loc, scale=self.scale)

    def draw_truncated(self, trun):
        if (self.a is not None):
            r = self.dist.cdf(trun, self.a, loc=self.loc, scale=self.scale)
            p = auxo.uniform_range([r, 1])
            return self.dist.ppf(p, self.a, loc=self.loc, scale=self.scale)
        else:
            r = self.dist.cdf(trun, loc=self.loc, scale=self.scale)
            p = auxo.uniform_range([r, 1])
            return self.dist.ppf(p, loc=self.loc, scale=self.scale)

    def fun(self, t, f='pdf'):
        # no switch/case in python...
        if (f == 'pdf'):
            if (self.a is not None):
                return self.dist.pdf(t, self.a, loc=self.loc, scale=self.scale)
            else:
                return self.dist.pdf(t, loc=self.loc, scale=self.scale)
        elif (f == 'cdf'):
            if (self.a is not None):
                return self.dist.cdf(t, self.a, loc=self.loc, scale=self.scale)
            else:
                return self.dist.cdf(t, loc=self.loc, scale=self.scale)
        elif (f == 'sf'):
            if (self.a is not None):
                return self.dist.sf(t, self.a, loc=self.loc, scale=self.scale)
            else:
                return self.dist.sf(t, loc=self.loc, scale=self.scale)
        elif (f == 'ppf'):
            if (self.a is not None):
                return self.dist.ppf(t, self.a, loc=self.loc, scale=self.scale)
            else:
                return self.dist.ppf(t, loc=self.loc, scale=self.scale)
        elif (f == 'moment'):
            if (self.a is not None):
                return self.dist.moment(t, self.a, loc=self.loc, scale=self.scale)
            else:
                return self.dist.moment(t, loc=self.loc, scale=self.scale)
        else:
            print('error')

    def params(self):
        if (self.a is not None):
            return self.reparam([self.a, self.scale])
        else:
            return self.reparam([self.scale])
