# code that detects neural pattern shifts based on hidden markov model
# Mar 8, 2025, Hayoung Song

# prerequisite
# step 1. download preprocessed fMRI data from https://openneuro.org/datasets/ds005658
# step 2. download Schaefer et al.'s (2018) parcellation atlas
# step 3. extract voxel activity time series using code_extractbold.py
# step 4. download brainiak toolbox from https://github.com/brainiak/brainiak and add to directory using sys.path.extend

directory_brainiak = '/folder'
directory_ts = '/folder'
directory_index = '/folder'
directory_hmm = '/folder'

import sys
import os
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import pickle
sys.path.extend(directory_brainiak)
from brainiak.brainiak.eventseg.event import EventSegment
np.random.seed(0)

def plot_tt_similarity_matrix(ax, run_BOLD, bounds, bounds_gth):
    ax.imshow(np.corrcoef(run_BOLD.T), cmap='viridis')

    bounds_aug = np.concatenate(([-1], bounds, [ev.segments_[0].shape[0]-1]))
    for i in range(len(bounds_aug) - 1):
        rect = patches.Rectangle(
            (bounds_aug[i]+0.5, bounds_aug[i]+0.5),
            bounds_aug[i + 1] - bounds_aug[i],
            bounds_aug[i + 1] - bounds_aug[i],
            linewidth=2, edgecolor='cyan', facecolor='none')
        ax.add_patch(rect)

    bounds_aug = np.concatenate(([-1], bounds_gth, [ev.segments_[0].shape[0]-1]))
    for i in range(len(bounds_aug) - 1):
        rect = patches.Rectangle(
            (bounds_aug[i]+0.5, bounds_aug[i]+0.5),
            bounds_aug[i + 1] - bounds_aug[i],
            bounds_aug[i + 1] - bounds_aug[i],
            linewidth=2, edgecolor='w', facecolor='none')
        ax.add_patch(rect)

def create_diagmask(mask):
    diag_mask = np.zeros_like(mask, dtype=bool)
    done=0
    while not done:
        for k in range(mask.shape[0]):
            d = np.diag(mask, k=k)
            if ~(d > 0).any():
                diag_limit = k
                done = 1
                break

    diffval = []
    for k in range(3, diag_limit+1):
        for ki in range(1, k):
            row_ix, col_ix = np.diag_indices_from(diag_mask)
            diag_mask[row_ix[:-ki], col_ix[ki:]] = True
        diffval.append([k, np.abs(np.sum(mask * diag_mask)-np.sum(~mask*diag_mask))])
    diffval=np.array(diffval)
    k = diffval[np.where(diffval[:,1]==np.min(diffval[:,1]))[0],0][0]

    diag_mask = np.zeros_like(mask, dtype=bool)
    for ki in range(1, k):
        row_ix, col_ix = np.diag_indices_from(diag_mask)
        diag_mask[row_ix[:-ki], col_ix[ki:]] = True
    return diag_mask, k

def interpolateMat(matrix):
    df = pd.DataFrame(matrix)
    df_interpolated = df.interpolate(method='linear', axis=1, limit_direction='both')
    return df_interpolated.to_numpy()


flist = {}
flist[1] = ['sub-1001', 'sub-1005', 'sub-1008', 'sub-1011', 'sub-1014', 'sub-1017', 'sub-1020', 'sub-1023', 'sub-1026', 'sub-1029', 'sub-1033', 'sub-1039']
flist[2] = ['sub-2006', 'sub-2009', 'sub-2012', 'sub-2015', 'sub-2018', 'sub-2021', 'sub-2024', 'sub-2027', 'sub-2034', 'sub-2038', 'sub-2040'] # 'sub-2030'
flist[3] = ['sub-3004', 'sub-3007', 'sub-3013', 'sub-3016', 'sub-3019', 'sub-3022', 'sub-3025', 'sub-3031', 'sub-3037', 'sub-3041'] # 'sub-3010', 'sub-3028'
tasklist = ['01','02','03','04','05','06','07','08','09','10']
nroi = 100 # schaefer et al.'s 100 cortical parcel
minTime = 3
nsubj = len(flist[1])+len(flist[2])+len(flist[3])


for roi in np.arange(1, nroi+1):
    if os.path.exists(directory_hmm+'/roi' + str(roi))==False:
        os.mkdir(directory_hmm+'/roi' + str(roi))

    for si, sbb in enumerate(flist[1]+flist[2]+flist[3]):
        print(sbb)

        # roi time series of a participant
        with open(directory_ts+'/'+str(sbb)+'_bold.pkl', 'rb') as f: ts = pickle.load(f)
        with open(directory_index+'/'+str(sbb)+'_index.pkl', 'rb') as f: index = pickle.load(f)
        sceneindx, runindx = index['sceneindx'], index['runindx']

        ts_roi = ts[roi]

        # run HMM
        hmm_id = {}
        for scc in range(1, 48+1):
            run_eventseg = True
            print(str(sbb)+' roi'+str(roi)+' scene'+str(scc))

            scene_BOLD = ts_roi[:,np.where(sceneindx==scc)[0]]
            print(str(scene_BOLD.shape))
            if np.any(np.isnan(scene_BOLD[0,:])):
                scene_BOLD = interpolateMat(scene_BOLD)
            if np.any(np.isnan(scene_BOLD[:,0])):
                scene_BOLD = np.delete(scene_BOLD, np.where(np.isnan(scene_BOLD[:,0]))[0], axis=0)
            if scene_BOLD.shape[1]<10:
                run_eventseg = False
            else:
                if scene_BOLD.shape[1]<20:
                    startK = 2
                else:
                    startK = 3
                wd = []
                for k in range(startK, int(np.floor(scene_BOLD.shape[1] / minTime)) + 1):
                    ev = EventSegment(k)
                    ev.fit(scene_BOLD.T)

                    w = np.zeros_like(ev.segments_[0])
                    w[np.arange(w.shape[0]), np.argmax(ev.segments_[0], axis=1)] = 1
                    mask = np.dot(w, w.T).astype(bool)

                    local_mask, optK = create_diagmask(mask)

                    within_vals = np.corrcoef(scene_BOLD.T)[mask * local_mask]
                    across_vals = np.corrcoef(scene_BOLD.T)[~mask * local_mask]
                    wd.append([k, np.mean(within_vals) - np.mean(across_vals)])
                wd = np.array(wd)

                '''
                f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,4.5))

                k = int(wd[np.argsort(wd[:,1])[-1], 0])
                ev = EventSegment(k)
                ev.fit(scene_BOLD.T)
                bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
                plot_tt_similarity_matrix(ax1, scene_BOLD, bounds, bounds)

                k = int(wd[np.argsort(wd[:,1])[-2], 0])
                ev = EventSegment(k)
                ev.fit(scene_BOLD.T)
                bounds = np.where(np.diff(np.argmax(ev.segments_[0], axis=1)))[0]
                plot_tt_similarity_matrix(ax2, scene_BOLD, bounds, bounds)

                ax3.plot(wd[:,0], wd[:,1])
                plt.suptitle('sub-'+str(sbb)+' scene'+str(scc))
                '''

                if np.all(np.diff(wd[:, 1]) < 0) or np.all(np.diff(wd[:, 1]) > 0):
                    run_eventseg = False

            if run_eventseg==True:
                k = int(wd[np.argsort(wd[:, 1])[-1], 0])
                ev = EventSegment(k)
                ev.fit(scene_BOLD.T)
                w = np.zeros_like(ev.segments_[0])
                w[np.arange(w.shape[0]), np.argmax(ev.segments_[0], axis=1)] = 1
                mask = np.dot(w, w.T).astype(bool)
                local_mask, optK = create_diagmask(mask)
                hmm_id[scc] = np.argmax(ev.segments_[0], axis=1)
            else:
                hmm_id[scc] = np.repeat([0], scene_BOLD.shape[1])

        f = open(directory_hmm+'/roi' + str(roi) + '/'+str(sbb)+'_hmmid.pkl', "wb")
        pickle.dump(hmm_id, f)
        f.close()
