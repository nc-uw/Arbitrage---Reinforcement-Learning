#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 16:57:39 2018

@author: nc57
"""

#from __future__ import division
import numpy as np
import math
np.random.seed(7)

## value of x when episode terminates
## reward function of a cart-pole
class battery():
    
    def __init__(self):
        self.Emax_mwh = 10.
        self.Emin_mwh = 0.0
        self.Deltamax_mw = 0.52
        self.Deltamin_mw = -0.52
        self.eff_ch = 1.
        self.eff_disch = 1.
        self.Et = self.Emin_mwh
        self.rate_ch = 0.5
        self.rate_disch = 0.5
        self.Edelta = 0.
        self.t_hour = 1.
        self.gamma = 1.0
        self.gamma_pow = 0.
        self.reward = 0
        self.returns = 0
        #self.A = []
        #self.S = [self.Et]

    def scale_state(self):
        return (self.Et - self.Emin_mwh)/(self.Emax_mwh - self.Emin_mwh)        

    def _delta_state(self, cmd):
        if cmd == 'charge':
            self.Edelta = 1 * self.rate_ch * self.t_hour * self.eff_ch
        elif cmd == 'discharge':
            self.Edelta = -1 * self.rate_disch * self.t_hour * self.eff_disch
        elif cmd == 'hold':
            self.Edelta = 0
    
    def update_state(self, cmd):
        #cmd = cmd
        self._delta_state(cmd)
        if (self.Et + self.Edelta) > self.Emax_mwh:
            '''
            print ('\nmax cap constraint violated..')
            print ('\n..cmd switched2 discharge')
            '''
            cmd = 'discharge'
            self._delta_state(cmd)
        elif (self.Et + self.Edelta) < self.Emin_mwh:
            '''
            print ('\nmin cap constraint violated..')
            print ('\n..cmd switched2 charge')
            '''
            cmd = 'charge'
            self._delta_state(cmd)
        self.Et += self.Edelta
        #self.A += [cmd]
        #self.S += [self.Et]
        return cmd
    
    def rewardfn(self, pr):
        disc = math.pow(self.gamma, self.gamma_pow)
        self.reward = -1. * pr * self.Edelta
        self.returns += disc * self.reward       
        self.gamma_pow += 1
        
    #def gen_episode(self, pr):
        