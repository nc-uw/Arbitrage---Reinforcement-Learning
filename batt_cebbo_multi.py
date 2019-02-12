#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 21:12:00 2018

@author: nc57
"""

#from __future__ import division
import numpy as np
np.random.seed(7)

class bbo():
    
    def __init__(self, N, K, Ke, state_size, action_space):
        self.state_size = state_size
        self.action_space = action_space
        self.action_dim = len(self.action_space)
        
        self.N = N
        self.K = K
        self.Ke = Ke
        self.epsilon = 1e-10
        
        #self.mu_init = np.zeros(self.state_size)
        # dim for mu_init: A * S (not S * A to accomadate generating W)
        self.mu_init = np.random.random((self.action_dim, self.state_size))
        self.cov_init = np.identity(self.state_size)*np.random.random()
        
        '''
        mu_init = np.random.random((3, 5))
        cov_init = np.identity(5)*np.random.random()
        a = 0        
        for k in range(K):
            b = 0
            for mi in mu_init:   
                if b == 0:
                    x = np.random.multivariate_normal(mi, cov_init, 1)
                else:
                    x = np.vstack([x, np.random.multivariate_normal(mi, cov_init, 1)])
                b += 1
            if a == 0:
                y = [x]
            else:
                y = np.vstack([y, [x]])
            a += 1
        W_init = y
        '''        
        
        a = 0        
        for k in range(self.K):
            b = 0
            for mi in self.mu_init:   
                if b == 0:
                    x = np.random.multivariate_normal(mi, self.cov_init, 1)
                else:
                    x = np.vstack([x, np.random.multivariate_normal(mi, self.cov_init, 1)])
                b += 1
            if a == 0:
                y = [x]
            else:
                y = np.vstack([y, [x]])
            a += 1
        self.W_init = y
        
        self.mu = self.mu_init
        self.cov = self.cov_init
        self.W = self.W_init
        
        
    def getW(self):
        #self above
        a = 0        
        for k in range(self.K):
            b = 0
            for mi in self.mu:   
                if b == 0:
                    x = np.random.multivariate_normal(mi, self.cov, 1)
                else:
                    x = np.vstack([x, np.random.multivariate_normal(mi, self.cov, 1)])
                b += 1
            if a == 0:
                y = [x]
            else:
                y = np.vstack([y, [x]])
            a += 1
        self.W = y
        
    
    def W2pol(self, w, state):
        #state = [batt.Et]
        Xs = np.transpose(np.array( [[1] + state] ))
        #Y = np.matmul(np.transpose(w),Xs)
        Y = np.matmul(w,Xs)
        #Y = np.array(Y, dtype=np.float128)
        #expY = np.exp(Y, dtype=np.float128)
        #probY = expY / np.sum(expY, axis=0)
        action_index = np.argmax(Y)
        action = self.action_space[action_index]
        return action
    
    
    def update_pol(self, Wfilt):        
        # notes:
        # mu can only be a vector
        # a. size of mu = size of R in S
        # b. size of mu = size of R in SXA
        self.mu = np.mean(Wfilt , axis=0)
        self.cov = self.epsilon * np.identity(self.state_size)
        for k in range(self.Ke):
            self.cov += np.matmul(np.transpose((Wfilt[k] - self.mu)), (Wfilt[k] - self.mu))
        self.cov /= (self.Ke + self.epsilon)         
        #self.mu = np.mean(Wfilt)
        
        
        
    
        