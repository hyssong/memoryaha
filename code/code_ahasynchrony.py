# compile fMRI participants' aha button presses and calculate pairwise synchrony
# Mar 8, 2025, Hayoung Song

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
import scipy
np.random.seed(1)

def smooth(aha):
    aha_smooth = aha.copy()
    for scc in range(1, 48+1):
        if np.sum(aha[scc]) > 0:
            id = np.concatenate((np.where(aha[scc] == 1)[0] - 1, np.where(aha[scc] == 1)[0], np.where(aha[scc] == 1)[0] + 1))
            if id[0] == -1: id = id[1:]
            if id[-1] >= len(aha[scc]): id = id[:-1]
            aha_smooth[scc][id] = 1
    return aha_smooth

def create_null_aha(ahaid):
    numaha, totalaha = 0, 0
    for scc in range(1, 48+1):
        numaha = numaha + np.sum(ahaid[scc])
        totalaha = totalaha + ahaid[scc].shape[0]
    tmp = np.zeros((totalaha,))
    tmp[np.random.permutation(totalaha)[:int(numaha)]] = 1
    ahaid_shuff = {}
    valid = 0
    for scc in range(1, 48+1):
        ahaid_shuff[scc] = tmp[valid:valid + ahaid[scc].shape[0]]
        valid = valid + ahaid[scc].shape[0]
    return ahaid_shuff

ahamoments_csv = pd.read_csv('/data/ahabutton.csv')
subjlist = np.sort(np.unique(ahamoments_csv['subject']))
scene_nTR = np.array(pd.read_csv('/data/sceneindex.csv')['nTR'])

aha_id = {}
for si, subj in enumerate(subjlist):
    # --------------------- run information --------------------- #
    groupid = int(str(subj)[0])
    run = np.array(pd.read_csv('/data/groupscene.csv')['run'])
    scene = np.array(pd.read_csv('/data/groupscene.csv')['g'+str(groupid)+'.sceneid'])
    character = np.array(pd.read_csv('/data/groupscene.csv')['g'+str(groupid)+'.char'])

    # --------------------- aha information --------------------- #
    idd = np.where(ahamoments_csv['subject'] == subj)[0]
    aha_run = np.array(ahamoments_csv['run'][idd])
    aha_scene = np.array(ahamoments_csv['scene'][idd])
    aha_sceneid = np.zeros((aha_scene.shape[0],))
    for i in range(aha_scene.shape[0]):
        aha_sceneid[i] = scene[np.where(run==aha_run[i])[0][aha_scene[i]-1]]
    aha_sceneid=np.array(aha_sceneid,dtype=int)
    aha_tr = np.array(ahamoments_csv['TR (scene)'][idd])

    aha_id[subj] = {}
    for scc in range(1, 48+1):
        nTR = scene_nTR[scc-1]
        aha_id[subj][scc] = np.zeros((nTR + 2,))
    for i in range(aha_sceneid.shape[0]):
        if aha_tr[i]>=len(aha_id[subj][aha_sceneid[i]]):
            print(str(subj)+' '+str(aha_tr[i])+' '+str(len(aha_id[subj][aha_sceneid[i]])))
        else:
            aha_id[subj][aha_sceneid[i]][aha_tr[i]] = 1
    for scc in range(1, 48+1):
        aha_id[subj][scc] = np.array(aha_id[subj][scc], dtype='int')


# --------------------- dice coefficient --------------------- #
niter = 10000 # chance distribution
dice_same, dice_diff, dice_real, dice_null = [], [], [], []
for s1 in range(len(subjlist)-1):
    for s2 in range(s1+1, len(subjlist)):
        aha_smooth_s1 = smooth(deepcopy(aha_id[subjlist[s1]]))
        aha_smooth_s2 = smooth(deepcopy(aha_id[subjlist[s2]]))
        aha_smooth_s1_cat, aha_smooth_s2_cat = [], []
        for scc in range(1, 48+1):
            if scc==1: aha_smooth_s1_cat = aha_smooth_s1[scc]
            else: aha_smooth_s1_cat = np.concatenate((aha_smooth_s1_cat, aha_smooth_s1[scc]))
            if scc==1: aha_smooth_s2_cat = aha_smooth_s2[scc]
            else: aha_smooth_s2_cat = np.concatenate((aha_smooth_s2_cat, aha_smooth_s2[scc]))
        dice = 2*np.sum(aha_smooth_s1_cat*aha_smooth_s2_cat) / (np.sum(aha_smooth_s1_cat) + np.sum(aha_smooth_s2_cat))
        dice_real.append(dice)
        if str(subjlist[s1])[0]==str(subjlist[s2])[0]:
            dice_same.append(dice)
        else:
            dice_diff.append(dice)

        dice_n = np.zeros((niter,))
        for iter in range(niter):
            aha_s1_shuff = smooth(create_null_aha(deepcopy(aha_id[subjlist[s1]])))
            aha_s2_shuff = smooth(create_null_aha(deepcopy(aha_id[subjlist[s1]])))
            aha_s1_shuff_cat, aha_s2_shuff_cat = [], []
            for scc in range(1, 48+1):
                if scc==1: aha_s1_shuff_cat = aha_s1_shuff[scc]
                else: aha_s1_shuff_cat = np.concatenate((aha_s1_shuff_cat, aha_s1_shuff[scc]))
                if scc==1: aha_s2_shuff_cat = aha_s2_shuff[scc]
                else: aha_s2_shuff_cat = np.concatenate((aha_s2_shuff_cat, aha_s2_shuff[scc]))
            dice_n[iter] = 2*np.sum(aha_s1_shuff_cat*aha_s2_shuff_cat) / (np.sum(aha_s1_shuff_cat) + np.sum(aha_s2_shuff_cat))
        dice_null.append(dice_n)

r = np.nanmean(dice_real,0)
n = np.nanmean(np.array(dice_null), 0)
z = (r-np.mean(n))/np.std(n)
r = r-np.mean(n)
n = n-np.mean(n)
p = (1+len(np.where(np.abs(r)<=np.abs(n))[0]))/(1+niter)
print('z='+str(np.round(z,5))+', p='+str(np.round(p, 5)))

fig, ax = plt.subplots(1,1)
ax.hist(dice_diff, color='#B3B3B3', bins=30)
ax.hist(dice_same, color='#676767', bins=30)
ax.vlines(np.mean(dice_same), 47, 48, color='red')
ax.vlines(np.mean(dice_diff), 47, 48)
ax.spines[['right', 'top']].set_visible(False)
ax.set_xticklabels([]), ax.set_yticklabels([])
print(scipy.stats.ranksums(dice_diff, dice_same))
