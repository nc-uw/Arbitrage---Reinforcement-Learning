#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 22:56:32 2018

@author: nc57
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from multiprocessing import Process #, Queue, current_process, freeze_support
import batt_caiso
import batt_env
import batt_cebbo_multi
np.random.seed(7)

####### EVALUATE: GENERATE EPISODES AS PER AGENT POLICY #######
def episode (w, P, batt, agent, price_state_size):
    A = []
    S = []
    for i in range(1, len(P)):
        if i <= price_state_size: 
            state = [batt.Et] + list(np.zeros(price_state_size-i)) + list(P[0:i])
        else:
            state = [batt.Et] + list(P[i-price_state_size:i])
        a = agent.W2pol(w, state)
        a = batt.update_state(a)
        #print('action', a)
        A += [a]
        S += [batt.Et]
        batt.rewardfn(P[i])
    return S, A, batt.returns
    
def evaluate(k, P, agent, price_state_size):
    w = agent.W[k]
    J = 0
    #T = []
    JN_all = []
    for n in range(agent.N):
        batt = batt_env.battery()
        S, A, returns = episode(w, P, batt, agent, price_state_size)
        J+=returns
        #print('terminal state encountered ..')
        #print('reward: ', G)
        JN_all += [returns]
    J=J/agent.N
    #print ('JN_all', JN_all)
    return (J, JN_all, A, S)

####### GENERATE (ONE) RANDOM EPISODE #######
def random_ep(P, CMD, CMD_prob):
    #call battery
    batt = batt_env.battery()
    c=0
    for pr in P:
        c+=1
        cmd = np.random.choice(CMD, 1, p=CMD_prob)[0]
        cmd = batt.update_state(cmd)
        batt.rewardfn(pr)
        print( '\nprice: {}, action: {}, state: {}, reward: {}, returns: {}'.\
              format(round(pr,2), cmd, batt.Et, round(batt.reward,2), round(batt.returns,2)) )
        
        
##### START #####
#gaussian price
'''
mean = 30
sigma = 15
P = np.random.normal(mean, sigma, 100)
'''

#caiso price
#P = batt_caiso.caiso_price('CAISO_201809.csv', show_info=False, show_plot=True)
P = pd.read_csv('Price2016.csv')
plt.plot(P)

#forecast price


#action space
CMD = ['charge', 'discharge', 'hold']
CMD_prob = [1/3, 1/3, 1/3]

# declare state sizes
price_state_size = 3
batt_state_size = 1
state_size = 1 + price_state_size + batt_state_size

#iterations (N)
itr = 50
tr_size = 552
plt.plot(P[tr_size:])
plt.plot(P[:tr_size])
#random ep test
#random_ep(P, CMD, CMD_prob)

####### TRAINING #######
#call agent
agent = batt_cebbo_multi.bbo(N=1, K=500, Ke=10, state_size=state_size, action_space = CMD)
plot_data = {}
plot_data_filt = {}
plot_action_filt = {}
plot_state_filt = {}
Jitr = []
Aitr = []
Sitr = []
for i in range(itr):
    #batt = batt_env.battery()
    #print ('i', i, 'mu', agent.mu, 'cov', agent.cov)
    agent.getW()
    eval_tup = [evaluate(k, P[tr_size:], agent, price_state_size) for k in range(agent.K)]
    #make below using zips
    J = np.array([e[0] for e in eval_tup])
    Jitr.append([e[1] for e in eval_tup])
    A = np.array([e[2] for e in eval_tup])
    Aitr.append([e[2] for e in eval_tup])
    S = np.array([e[3] for e in eval_tup])
    Sitr.append([e[3] for e in eval_tup])
    #filter beta and J
    indices = np.argsort(J)[len(J)-agent.Ke:]
    Jfilt = J[indices]
    Afilt = A[indices]
    Sfilt = S[indices]
    print('i', i) 
    print('Jfilt', Jfilt)
    filt = np.zeros(agent.K, dtype=bool)
    for i in list(indices): filt[i] = 1
    Wfilt = agent.W[filt]
    #print (Wfilt)
    agent.update_pol(Wfilt)
    plot_data[i]= J
    plot_data_filt[i]= Jfilt
    plot_action_filt[i]= Afilt
    plot_state_filt[i]= Sfilt

####### VALIDATION #######
eval_tup = [evaluate(k, P[:tr_size], agent, price_state_size) for k in range(agent.K)]
#make below using zips
J = np.array([e[0] for e in eval_tup])
Jitr.append([e[1] for e in eval_tup])
A = np.array([e[2] for e in eval_tup])
Aitr.append([e[2] for e in eval_tup])
S = np.array([e[3] for e in eval_tup])
Sitr.append([e[3] for e in eval_tup])
#filter beta and J
indices = np.argsort(J)[len(J)-agent.Ke:]
Jfilt = J[indices]
Afilt = A[indices]
Sfilt = S[indices]
#Jfilt_val = Jfilt
print('i', i) 
print('Jfilt', Jfilt)

'''
plt.plot(P)
plt.plot(plot_action_filt[48][0])
plt.plot(plot_state_filt[48][0])
'''
    
'''
np.save('plot_data_trailv2_2.npy', plot_data)
np.save('plot_data_filt_trailv2_2.npy', plot_data_filt)
np.save('plot_action_filt_trailv2_2.npy', plot_action_filt)
np.save('plot_state_filt_trailv2_2.npy', plot_state_filt)
Jitr = np.ravel(Jitr)
np.save('Jitr_trailv2_1.npy', Jitr)
np.save('A_trail_17.npy', Aitr)
np.save('Price_trailv2_1.npy', P)
print('Jitr', Jitr)
return Jitr, plot_data
'''


# trail 1,2,3,4
# K=500, Ke=20, ep=1-8
# trail 5
# K=500, Ke=100, ep=1-10
# trail 6, 7
# K=100, Ke=30, ep=1-10
# trail 8, 9, 10, 11, 12, 13, 14, 15, 16
# K=150, Ke=30, ep=1-10

'''
#############################################################
########################Experiments#########################
def trailrun(trails, itr, Ke, env):
    Jitr_all = []
    plot_data_all = {}
    for i in range(trails):
        Jitr, plot_data = env.experiments(itr, agent.Ke)
        Jitr_all.append(Jitr)
        plot_data_all[i] = plot_data
    np.save("Jitr_all.npy", Jitr_all)
    np.save("plot_data.npy", plot_data)
    return Jitr_all

def worker():
    Ke = 1
    K = 2
    N = 3
    trails = 4
    itern = 5
    cart = cartpole(N,K)
    Jitr_all = trailrun(trails, itern, Ke, cart)        
    proc = os.getpid()
    np.save(str(proc)+".npy", Jitr_all)


if __name__ == '__main__':
    jobs = []
    for i in range(2):
        p = Process(target=worker)
        #p.start()
        jobs.append(p)
    
    for z in jobs:
        np.random.seed()
        z.start()       
'''

        