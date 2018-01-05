import numpy as np
import pandas as pd
import random as rnd

#functions - not required
def per(n):
    for i in range(1<<n):
        s=bin(i)[2:]
        s='0'*(n-len(s))+s
        print map(int,list(s))
        
def arb(n,mwh):
    a=np.zeros((n,n))
    for i in range(0,n):
        a[i][:i+1]=mwh
    return(a)

def costold(a,n):
    cost=[];
    for i in range(0,h-n):
        cost.append(np.dot(a,p[i:i+n]))
    return (cost)

#price
p=np.array([35.33,31.36,32.27,32.35,30.80,33.87,43.19,48.24,43.47,42.13,39.28,37.35,34.77,33.20,31.39,31.54, 35.84,47.29,45.17,39.98,35.65,34.07,34.32,32.66])
h=24

#battery params
dc_len=3
c_len=3
charge_mwh=0.5
discharge_mwh=-0.5
charge_cap=0.5*c_len
discharge_cap=0.5*dc_len
min_charge=0;
max_charge=charge_cap
batt_init=0
eff=1
#batt=dict([("charge length",c_len), ("discharge length",dc_len), ("") ])

#functions 
def cost(t,pos):
    cost=actions[pos]*p[t]
    return (cost)

#actions init
ac_q=np.zeros(h)
ad_q=np.zeros(h)
an_q=np.zeros(h)
actions=np.array([charge_mwh,discharge_mwh,0])
options=[0,1,2]

#RL datasets init
q_value=np.column_stack((ac_q, ad_q, an_q))
q_current=[]
q_next=[]
C_iter=[]

#RL params init
alpha=0.01
gamma=1
lmbd=1
itr=10

for i in range(0,itr):        
    batt_state=batt_init;
    c=[]
    pos=[]
    
    for t in range(0,h):
        if batt_state==min_charge:
            options=[0,2]
        if batt_state==max_charge:
            limits=[1,2]
        if i==0:
            pos.append(rnd.choice(options))
        else:
            temp=np.array([ii for ii,x in enumerate(q_value[t]) if x == min(q_value[t][options])])
            pos.append(rnd.choice(temp))
        batt_state=batt_state+actions[pos[t]]
    
    #cost computation
    c=actions[pos]*p
    q_current=q_value[np.arange(len(q_value)), pos]
    q_next=np.append(q_current[1:h], 0)
    del_q=alpha*((c+lmbd*q_next)-q_current)
    q_value[np.arange(len(q_value)), pos]=q_value[np.arange(len(q_value)), pos]+del_q
    C=c[::-1].cumsum()[::-1]
    C_iter.append(C[1])
