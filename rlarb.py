
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 11 18:45:41 2017

@author: scsingh
"""
import numpy as np
import pandas as pd
import random as rnd
import matplotlib as plt1
import matplotlib.pyplot as plt2

#price
p=np.array([35.33,31.36,32.27,32.35,30.80,33.87,43.19,48.24,43.47,42.13,39.28,37.35,34.77,33.20,31.39,31.54, 35.84,47.29,45.17,39.98,35.65,34.07,34.32,32.66])
p=np.array([30.96,28.13,26.62,26.16,26.86,33.25,47.40,54.88,57.92,63.15,65.45,62.44,64.01,57.15,52.34,54.54,66.40,101.38,97.82,80.88,76.48,63.21,55.83,45.96]);
#p=np.array([35.33],[31.36],[32.27],[32.35],[30.80],[33.87],[43.19],[48.24],[43.47],[42.13],[39.28],[37.35],[34.77],[33.20],[31.39],[31.54],[35.84],[47.29],[45.17],[39.98],[35.65],[34.07],[34.32],[32.66])
h=24

#battery params
eff=1
dc_len=5
c_len=5
charge_mwh=0.27
discharge_mwh=-0.27
min_charge=0.0;
max_charge=3.0;
#batt=dict([("charge length",c_len), ("discharge length",dc_len), ("") ])

#battery datasets init
normal_options = ([charge_mwh], [0.0], [discharge_mwh])
min_options = ([charge_mwh], [0.0], [0.0])
max_options = ([discharge_mwh], [0.0], [0.0])
a=(normal_options, min_options, max_options)
r=(normal_options*p, min_options*p, max_options*p)

#RL datasets init
q=np.zeros(np.shape(r)) 
q_current=np.zeros(np.shape(r))
q_next=np.zeros(np.shape(r))

#RL params init
alpha=0.1
gamma=1
lmbd=1
itr=5000
epsln=0.1
tot_rew=[]

batt_init=min_charge

for i in range(0,itr):        
    batt_state=[(batt_init)]
    rew=[]
    trk = [('pos_q, choice')]
    print "ïteration", i+1, "initiated"
    for t in range(0,h):
        if batt_state[t]+discharge_mwh <= min_charge: 
            pos_q = 1 
            options=1
            print "time", t, "charge_now", batt_state[t],
        elif batt_state[t]+charge_mwh >= max_charge: 
            pos_q = 2 
            options = 1 
            print "time", t, "charge_now", batt_state[t],
        else: 
            pos_q = 0 
            options=2
            print "time", t, "charge_now", batt_state[t],
        #print "time step", t, "charge_now", batt_state[t], "inital charge_now", batt_init
        #print "time step", t, "position in q table", pos_q, "options avaible ", options
        if (i==0 or rnd.uniform(0,1)<epsln):
            choice=rnd.choice(range(options+1))
            #calculate reward
            rew.append(-1*p[t]*a[pos_q][choice][0])
            trk.append((pos_q,choice))
            print "random action",
            #print "time step", t, "random choice", choice, "position", trk[t+1]
        else:
            evalz=[]
            for j in range(options+1):
                evalz.append(q[pos_q][j][t])                
            temp=np.array([ii for ii,x in enumerate(evalz) if x == max(evalz)])
            choice=(rnd.choice(temp))
            rew.append(-1*p[t]*a[pos_q][choice][0])
            trk.append((pos_q,choice))
            print "greedy action",
            #print "max action reward", rew[t]
        batt_state.append(batt_state[t]+a[pos_q][choice][0])
        print "charge_next", batt_state[t+1], "reward", rew[t]
        #batt_state.pop()
    print "end of episode!", "total reward", sum(rew)
    tot_rew.append(sum(rew))
    #print "total reward", total_rew

    #q_update    
    print "updating q table!"
    for t in range(0,h-1):                
        q[trk[t+1][0]][trk[t+1][1]][t]+=alpha*((rew[t]+lmbd*q[trk[t+2][0]][trk[t+2][1]][t+1])-q[trk[t+1][0]][trk[t+1][1]][t])
        #print t,
    q[trk[t+1][0]][trk[t+1][1]][t]+=alpha*((rew[t]+0)-q_current[pos_q][choice][t])
    #print h
    print "updated!" 
    print "iteration", i+1, "completed"
    print "***"
    print "**"
    print "*"
    print " "

#plot schedule vs price
x = range(1,25)
y1 = np.array(batt_state)
y2 = p
fig, ax1 = plt2.subplots()
ax2 = ax1.twinx()
ax1.plot(x, y1, 'g-')
ax2.plot(x, y2, 'b-')
ax1.set_xlabel('Hour')
ax1.set_ylabel('battery charge', color='g')
ax2.set_ylabel('price', color='b')
plt2.title('charge/discharge schedule against price')
plt2.show()
plt2.grid(True)

#plot cost vs iteration
plt2.plot(range(len(tot_rew)),np.array(tot_rew))
plt2.xlabel('iterations')
plt2.ylabel('reward of trajectory')
plt2.title('change in rewards by acting greedily based on updated values of q(s,a) table')
plt2.grid(True)
