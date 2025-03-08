# extracting voxel time series after applying a parcel mask to preprocessed EPIs
# Nov 24, 2024, Hayoung Song

# preprocessed fMRI data can be downloaded from: https://openneuro.org/datasets/ds005658

import numpy as np
from nilearn.image import load_img
import matplotlib.pyplot as plt
import scipy.stats
import pandas as pd

def niftimask(nroi_cor, nroi_sub, directory):
    cortical = directory+'/template/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-02_atlas-Schaefer2018_desc-'+str(nroi_cor)+'Parcels17Networks_dseg.nii.gz'
    if nroi_sub==16: subcortical = directory+'/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S1_3T_2009cAsym.nii.gz'
    elif nroi_sub==32: subcortical = directory+'/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S2_3T_2009cAsym.nii.gz'
    elif nroi_sub == 50: subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S3_3T_2009cAsym.nii.gz'
    elif nroi_sub == 54: subcortical = directory + '/template/Tian2020MSA_v1.1_3T_Subcortex-Only/Tian_Subcortex_S4_3T_2009cAsym.nii.gz'
    mask_cor = load_img(cortical).dataobj[:]
    mask_sub = load_img(subcortical).dataobj[:]

    for i1 in range(mask_sub.shape[0]):
        for i2 in range(mask_sub.shape[1]):
            for i3 in range(mask_sub.shape[2]):
                if mask_sub[i1,i2,i3]>0:
                    mask_sub[i1,i2,i3] = mask_sub[i1,i2,i3] + nroi_cor

    id = np.where(np.multiply(mask_cor, mask_sub)>0)
    mask = mask_cor + mask_sub
    mask[id[0],id[1],id[2]] = 0
    return mask

''' setting '''
flist = {}
flist[1] = ['sub-1001', 'sub-1005', 'sub-1008', 'sub-1011', 'sub-1014', 'sub-1017', 'sub-1020', 'sub-1023', 'sub-1026', 'sub-1029', 'sub-1033', 'sub-1039']
flist[2] = ['sub-2006', 'sub-2009', 'sub-2012', 'sub-2015', 'sub-2018', 'sub-2021', 'sub-2024', 'sub-2027', 'sub-2034', 'sub-2038', 'sub-2040'] # 'sub-2030'
flist[3] = ['sub-3004', 'sub-3007', 'sub-3013', 'sub-3016', 'sub-3019', 'sub-3022', 'sub-3025', 'sub-3031', 'sub-3037', 'sub-3041'] # 'sub-3010', 'sub-3028'
tasklist = ['01','02','03','04','05','06','07','08','09','10']
# sub-2030, sub-3010, sub-3028: large head motion participants
# sub-1023 task-03: only movie watching portion was recorded
nsubj = len(flist[1])+len(flist[2])+len(flist[3])

directory = '/foldername'

nroi_cor, nroi_sub = 100, 16
hrf = 4
mask = niftimask(nroi_cor, nroi_sub, directory)

# roi, groupid, si, subname, ti, task = 1, 1, 0, flist[1][0], 0, tasklist[0]
for groupid in range(1, 3+1):
    run = np.array(pd.read_csv(directory+'/socialaha-fMRI/socialaha_groupscene.csv')['run'])
    scene = np.array(pd.read_csv(directory+'/socialaha-fMRI/socialaha_groupscene.csv')['g'+str(groupid)+'.sceneid'])

    for si, subname in enumerate(flist[groupid]):
        # parcel mask is multiplied by each participant's brain mask applied during preprocessing
        submask = load_img(directory+'/masks/'+subname+'/'+subname+'_combined.nii.gz').dataobj[:]
        submask = np.multiply(submask, mask)

        for ti, task in enumerate(tasklist):
            print(subname+' task-'+task)

            # time stamps
            tst = pd.read_csv(directory+'/bids/'+subname+'/func/'+subname+'_task-'+task+'_events.tsv', sep='\t')
            tst['offset'] = tst['onset'] + tst['duration']
            tst['onset'] = tst['onset'] + hrf
            tst['offset'] = tst['offset'] + hrf

            # normalized BOLD time series of all voxels corresponding to each of the cortical & subcortical parcels
            epi = load_img(directory + '/derivatives/'+subname+'/'+subname+'_task-'+task+'_smoothed.nii.gz').dataobj[:]
            roi_ts = {}
            for roi in range(1, nroi_cor+nroi_sub+1):
                mask_id = np.where(submask == roi)
                ts = np.array([])
                ts = epi[mask_id[0], mask_id[1], mask_id[2], :]

                # remove time steps that were censored due to motion
                for tr in range(ts.shape[1]):
                    if np.all(ts[:,tr]==0):
                        ts[:,tr] = np.nan

                # normalize each voxel time series
                ts = scipy.stats.zscore(ts, axis=1, ddof=1, nan_policy='omit')

                roi_ts[roi] = ts

            # example
            # roi_ts[1][:,int(tst['onset'][0]):int(tst['offset'][0])]            # time series of roi=1 during first-event movie watching
            # roi_ts[50][:,int(tst['onset'][5]):int(tst['offset'][5])].shape     # time series of roi=50 during first-event aha explanation
            # roi_ts[50][:,int(tst['onset'][10]):int(tst['offset'][10])].shape   # time series of roi=50 during first-character impression
