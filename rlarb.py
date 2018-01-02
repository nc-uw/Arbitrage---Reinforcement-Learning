import numpy as np
import pandas as pd
import random as rnd

p=np.array([35.33,31.36,32.27,32.35,30.80,33.87,43.19,48.24,43.47,42.13,39.28,37.35,34.77,33.20,31.39,31.54, 35.84,47.29,45.17,39.98,35.65,34.07,34.32,32.66])
h=24

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

#mdp properties
options=3
#batt=dict([("charge length",c_len), ("discharge length",dc_len), ("") ])

#binary permutation
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

ac=arb(c_len, charge_mwh)
ad=arb(dc_len, discharge_mwh)

#cost permutation
def cost(a,n):
    cost=[];
    for i in range(0,h-n):
        cost.append(np.dot(a,p[i:i+n]))
    return (cost)
 
#initialization    
ac_cost=cost(ac,c_len)
ad_cost=cost(ad,dc_len)

ac_cost=np.dot(ac[1][1],p)
ad_cost=np.dot(ad[1][1],p)
an_cost=np.zeros((24,1))
cost=np.column_stack((ac_cost, ad_cost, an_cost))

ac_q=np.zeros(h)
ad_q=np.zeros(h)
an_q=np.zeros(h)
q_value=np.column_stack((ac_q, ad_q, an_q))

actions=np.array([charge_mwh,discharge_mwh,0])

#functions 
def cost(t,pos):
    cost=actions[pos]*p[t]
    return (cost)

#RL init
alpha=0.01
gamma=1
for i in range(0,1000):
    batt_state=batt_init;
    

C=[]
post=[]

for t in range(0,h):
    if i==0:
        pos=rnd.randint(0, len(actions)-1)
    else:
        pos=[ii for ii,x in enumerate(q_value[t]) if x == min(q_value[t])]
        if len(np.array([pos])) > 1:
            pos=rnd.choice(pos)
    #cost computation
    c=cost(t,pos)
    ct.append(c)
    post.append(pos)
c=actions[post]*p
C=np.cumsum(list(reversed(ct)))
print ct    
print C
print post