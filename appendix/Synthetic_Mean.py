import numpy as np
import json
import matplotlib.pyplot as plt
import sys
import pathlib
import os

current_directory = pathlib.Path.cwd()
print("Current directory:", current_directory)

SWD_directory = pathlib.Path(os.path.join(current_directory,"methods","python"))
print(SWD_directory)

sys.path.append(str(SWD_directory))

from SWD.Detector_TSWD import Detector as p_Detector
from SWD.utils import *

np.random.seed(0)
n_cps = 5
min_l = 45
max_l = 80
pos_l = np.arange(min_l,max_l+1,1)
run_lenght = np.random.choice(pos_l,n_cps)
cp_loc = np.cumsum(run_lenght)
print(run_lenght)
print(cp_loc)
nb_f = 3
#change in mean from 0 to 1
cov = np.eye(nb_f)
cov[0,0] = 0.1
cov[1,1] = 0.3
cov[2,2] = 0.4

mean = np.zeros(nb_f)

Data = np.random.multivariate_normal(mean,cov,size=run_lenght[0])

for n in run_lenght[1:]:
    #if np.sum(mean) == 0:
    mean = mean + [0,0,np.random.choice([-3,-2,2,3])]
    #else:
       # mean = np.zeros(nb_f)
    print()
    Data = np.vstack((Data,np.random.multivariate_normal(mean,cov,size=n)))
    mean = np.zeros(nb_f)



N = cp_loc[-1]

# fig, ax = plt.subplots(nb_f,1, figsize=(4,6))
# for i in range(nb_f):
#     ax[i].plot(range(N),Data[:,i], c="black", alpha=0.8)
#     for cp in cp_loc[:-1]:
#         ax[i].axvline(cp,ls='--',c='blue')


# ax[0].set_xlabel('time');
# ax[1].set_xlabel('time');
# ax[2].set_xlabel('time');
# ax[0].title.set_text('First component')
# ax[1].title.set_text('Second component')
# ax[2].title.set_text('Third component')
# fig.tight_layout()

annotations = {"1": cp_loc[:-1]}
Data_normalized = (Data- np.mean(Data))/np.std(Data)
SWATCH = p_Detector(Data,K=4,eps=1.3,kappa=3,mu=5,L=1000,p=2,fimp=True)
cps = SWATCH.run()
print(annotations)
print(cps)
print(SWATCH.evaluate(annotations,cps))


N = cp_loc[-1]
fig, ax = plt.subplots(nb_f,1, figsize=(4,6))
for i in range(nb_f):
    ax[i].plot(range(N),Data_normalized[:,i], c="black", alpha=0.8)
    for cp  in cp_loc[:-1]:
        ax[i].axvline(cp,color="blue")
    for cpd in cps:
        ax[i].axvline(cpd, color="red",ls="--")

ax[0].set_xlabel('time');
ax[1].set_xlabel('time');
ax[2].set_xlabel('time');
ax[0].title.set_text('First component')
ax[1].title.set_text('Second component')
ax[2].title.set_text('Third component')
fig.tight_layout();
#fig.savefig(os.path.join(current_directory,"appendix","Synthetic_Mean_1.pdf"))


ind_q = SWATCH.Q['i']
streched_q = []
streched_fimp = {'1':[],'2':[],'3':[]}
i_prev = 0
for q_ind , i_q in enumerate(ind_q):
    for i in range(i_prev,i_q):
        streched_q.append(SWATCH.Q['q'][q_ind])
        streched_fimp['1'].append(SWATCH.fimp_data['values'][q_ind][0])
        streched_fimp['2'].append(SWATCH.fimp_data['values'][q_ind][1])
        streched_fimp['3'].append(SWATCH.fimp_data['values'][q_ind][2])
    i_prev = i_q


obs = len(streched_fimp['1'])
fig2, ax2 = plt.subplots(nb_f,1,figsize=(6,6))
for i,val in streched_fimp.items():
    ax2[int(i)-1].plot(range(obs),val,color='black',label="MPC")
    ax2[int(i)-1].plot(range(obs),streched_q,color='grey',label="p")
    for cp  in cp_loc[:-1]:
        ax2[int(i)-1].axvline(cp,color="blue")
    for cpd in cps:
        ax2[int(i)-1].axvline(cpd, color="red",ls="--")

ax2[0].legend();
ax2[0].set_xlabel('time');
ax2[1].set_xlabel('time');
ax2[2].set_xlabel('time');
fig2.savefig(os.path.join(current_directory,"appendix","Synthetic_Mean_2.pdf"))



