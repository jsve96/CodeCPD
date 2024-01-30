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

Dataset_path = os.path.join(current_directory,"datasets","occupancy.json")
data, mat = load_dataset(Dataset_path)
with open(os.path.join(current_directory,'datasets','annotations.json'),'r') as file:
    META_ANNOTATIONS = json.load(file)

annotations = META_ANNOTATIONS['occupancy']
cp_loc = [anno for _,anno in annotations.items()]
cp_loc = [x for xs in cp_loc for x in xs]
cp_loc = list(set(cp_loc))
print(cp_loc)


np.random.seed(7)
SWATCH = p_Detector(mat,K=7,eps=1.2,kappa=2,mu=5,L=100,p=2,fimp=True)
cps = SWATCH.run()
print(annotations)
print(cps)
print(SWATCH.evaluate(annotations,cps))

ind_q = SWATCH.Q['i']
streched_q = []
streched_fimp = {'1':[],'2':[],'3':[],'4':[]}
i_prev = 0
for q_ind , i_q in enumerate(ind_q):
    for i in range(i_prev,i_q):
        streched_q.append(SWATCH.Q['q'][q_ind])
        streched_fimp['1'].append(SWATCH.fimp_data['values'][q_ind][0])
        streched_fimp['2'].append(SWATCH.fimp_data['values'][q_ind][1])
        streched_fimp['3'].append(SWATCH.fimp_data['values'][q_ind][2])
        streched_fimp['4'].append(SWATCH.fimp_data['values'][q_ind][3])
    i_prev = i_q

N = cp_loc[-1]
fig, ax = plt.subplots(4,2, figsize=(15,12))
for i in range(4):
    ax[i,0].plot(range(mat.shape[0]),mat[:,i], c="black", alpha=0.8)
    for cp  in cp_loc[:-1]:
        ax[i,0].axvline(cp,color="blue",alpha = 0.4)
    ax[i,0].axvline(cp_loc[-1],color="blue",alpha=0.4,label="ground truth")
    for cpd in cps[:-1]:
        ax[i,0].axvline(cpd, color="red",ls="--",alpha=0.6)
    ax[i,0].axvline(cps[-1],color="red",ls="--",alpha=0.6,label="annotation")

ax[0,0].legend();
ax[0,0].set_xlabel('time');
ax[1,0].set_xlabel('time');
ax[2,0].set_xlabel('time');
ax[3,0].set_xlabel('time');
ax[0,0].title.set_text('First component')
ax[1,0].title.set_text('Second component')
ax[2,0].title.set_text('Third component')
ax[3,0].title.set_text('Fourth component')

obs = len(streched_fimp['1'])

for i,val in streched_fimp.items():
    ax[int(i)-1,1].plot(range(obs),val,color='black',label="MPC")
    ax[int(i)-1,1].plot(range(obs),streched_q,color='grey',label="p")
    # for cp  in cp_loc:
    #     ax[int(i)-1,1].axvline(cp,color="blue",alpha=0.4)
    for cpd in cps:
        ax[int(i)-1,1].axvline(cpd, color="red",ls="--",alpha=0.6)

ax[0,1].set_xlabel('time');
ax[0,1].legend();
ax[1,1].set_xlabel('time');
ax[2,1].set_xlabel('time');
ax[3,1].set_xlabel('time');
ax[0,1].title.set_text('First component')
ax[1,1].title.set_text('Second component')
ax[2,1].title.set_text('Third component')
ax[3,1].title.set_text('Fourth component')


fig.tight_layout(); 
print(current_directory)
fig.savefig(os.path.join(current_directory,"appendix","occupancy_plot.pdf"))


# N = cp_loc[-1]
# fig, ax = plt.subplots(4,2, figsize=(10,12))
# for i in range(4):
#     ax[i,0].plot(range(mat.shape[0]),mat[:,i], c="black", alpha=0.8)
#     for cp  in cp_loc[:-1]:
#         ax[i,0].axvline(cp,color="blue",alpha = 0.4)
#     ax[i,0].axvline(cp_loc[-1],color="blue",alpha=0.4,label="ground truth")
#     for cpd in cps[:-1]:
#         ax[i,0].axvline(cpd, color="red",ls="--",alpha=0.6)
#     ax[i,0].axvline(cps[-1],color="red",ls="--",alpha=0.6,label="annotation")

# ax[0,0].legend();
# ax[0,0].set_xlabel('time');
# ax[1,0].set_xlabel('time');
# ax[2,0].set_xlabel('time');
# ax[3,0].set_xlabel('time');
# ax[0,0].title.set_text('First component')
# ax[1,0].title.set_text('Second component')
# ax[2,0].title.set_text('Third component')
# ax[3,0].title.set_text('Fourth component')

# obs = len(streched_fimp['1'])
# #fig2, ax2 = plt.subplots(4,1,figsize=(12,6))
# for i,val in streched_fimp.items():
#     ax[int(i)-1,1].plot(range(obs),val,color='black',label="MPC")
#     ax[int(i)-1,1].plot(range(obs),streched_q,color='grey',label="p")
#     # for cp  in cp_loc:
#     #     ax[int(i)-1,1].axvline(cp,color="blue",alpha=0.4)
#     for cpd in cps:
#         ax[int(i)-1,1].axvline(cpd, color="red",ls="--",alpha=0.6)
# ax[0,1].set_xlabel('time');
# ax[0,1].legend();
# ax[1,1].set_xlabel('time');
# ax[2,1].set_xlabel('time');
# ax[3,1].set_xlabel('time');
# ax[0,1].title.set_text('First component')
# ax[1,1].title.set_text('Second component')
# ax[2,1].title.set_text('Third component')
# ax[3,1].title.set_text('Fourth component')


# fig.tight_layout();
# print(current_directory)
# fig.savefig(os.path.join(current_directory,"appendix","bee_waggle.pdf"))
