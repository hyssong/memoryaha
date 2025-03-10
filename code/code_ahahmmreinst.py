# relationship between neural reinstatement, neural pattern shifts, and behavioral retrieval
# Mar 8, 2025, Hayoung Song

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def calculate_reinst_shift_retr(reinstatement, shift, retrieval, startT, endT, shift_id, retrieval_id):
    shift_retr = np.where((shift == shift_id) & (retrieval == retrieval_id))[0]
    if len(shift_retr)==0: reinst_shift_retr = np.repeat(np.nan, endT-startT+1)
    elif len(shift_retr)==1: reinst_shift_retr = reinstatement[shift_retr, :]
    else: reinst_shift_retr = np.nanmean(reinstatement[shift_retr, :], 0)
    return reinst_shift_retr

# data load & process
data = pd.read_csv('/data/summarydata_ahahmmreinst.csv')
subject = np.array(np.unique(data['subject']), dtype='int')
roi_sig = np.array(np.unique(data['parcel']), dtype='int')
startT, endT = -10, 3


# --------------------- figure 4A --------------------- #
reinst_shift1_retr1, reinst_shift1_retr0, reinst_shift0_retr1, reinst_shift0_retr0 = np.zeros((endT-startT+1, len(roi_sig), len(subject))), np.zeros((endT-startT+1, len(roi_sig), len(subject))), np.zeros((endT-startT+1, len(roi_sig), len(subject))), np.zeros((endT-startT+1, len(roi_sig), len(subject)))
for ri, roi in enumerate(roi_sig):
    for sbi, sbb in enumerate(subject):
        retrieval = np.array(data[(data['subject'] == sbb) & (data['parcel'] == roi)])[:,2]
        reinstatement = np.array(data[(data['subject'] == sbb) & (data['parcel'] == roi)])[:,3:17]
        shift = np.array(data[(data['subject'] == sbb) & (data['parcel'] == roi)])[:,17:31]
        shift = shift[:,np.where(np.arange(startT, endT+1)==-2)[0][0]]

        reinst_shift1_retr1[:, ri, sbi] = calculate_reinst_shift_retr(reinstatement, shift, retrieval, startT, endT, 1, 1)
        reinst_shift1_retr0[:, ri, sbi] = calculate_reinst_shift_retr(reinstatement, shift, retrieval, startT, endT, 1, 0)
        reinst_shift0_retr1[:, ri, sbi] = calculate_reinst_shift_retr(reinstatement, shift, retrieval, startT, endT, 0, 1)
        reinst_shift0_retr0[:, ri, sbi] = calculate_reinst_shift_retr(reinstatement, shift, retrieval, startT, endT, 0, 0)

plt.figure()
ax = plt.subplot(111)
ax.plot(np.arange(startT, endT+1), conv_z2r(np.nanmean(np.nanmean(reinst_shift1_retr1, 1), 1)), color='#009444')
ax.plot(np.arange(startT, endT+1), conv_z2r(np.nanmean(np.nanmean(reinst_shift1_retr0, 1), 1)), color='#009444', linestyle='dashed')
ax.plot(np.arange(startT, endT+1), conv_z2r(np.nanmean(np.nanmean(reinst_shift0_retr1, 1), 1)), color='#D75F4C')
ax.plot(np.arange(startT, endT+1), conv_z2r(np.nanmean(np.nanmean(reinst_shift0_retr0, 1), 1)), color='#D75F4C', linestyle='dashed')
ax.spines[['right', 'top']].set_visible(False)
plt.yticks([-0.01, 0, 0.01, 0.02, 0.03])
ax.set_xticklabels([]), ax.set_yticklabels([])
ax.tick_params(axis='both', which='major', length=6)


# --------------------- figure 4B --------------------- #
reinst_shift1, reinst_shift0, reinst_retr1, reinst_retr0 = np.zeros((len(roi_sig), len(subject)))+np.nan, np.zeros((len(roi_sig), len(subject)))+np.nan, np.zeros((len(roi_sig), len(subject)))+np.nan, np.zeros((len(roi_sig), len(subject)))+np.nan
for ri, roi in enumerate(roi_sig):
    for sbi, sbb in enumerate(subject):
        retrieval = np.array(data[(data['subject'] == sbb) & (data['parcel'] == roi)])[:,2]
        reinstatement = np.array(data[(data['subject'] == sbb) & (data['parcel'] == roi)])[:,3:17]
        reinstatement = np.nanmean(reinstatement[:,np.where(np.arange(startT, endT+1)==-7)[0][0]: np.where(np.arange(startT, endT+1)==-3)[0][0]+1], 1)
        shift = np.array(data[(data['subject'] == sbb) & (data['parcel'] == roi)])[:,17:31]
        shift = shift[:,np.where(np.arange(startT, endT+1)==-2)[0][0]]

        if len(np.where(shift==1)[0])>0: reinst_shift1[ri, sbi] = np.nanmean(reinstatement[np.where(shift==1)[0]])
        if len(np.where(shift==0)[0])>0: reinst_shift0[ri, sbi] = np.nanmean(reinstatement[np.where(shift==0)[0]])
        if len(np.where(retrieval==1)[0])>0: reinst_retr1[ri, sbi] = np.nanmean(reinstatement[np.where(retrieval==1)[0]])
        if len(np.where(retrieval==0)[0])>0: reinst_retr0[ri, sbi] = np.nanmean(reinstatement[np.where(retrieval==0)[0]])

# neural reinstatement & neural pattern shift
df = pd.DataFrame({'shift': np.concatenate((np.repeat(1, len(subject)), np.repeat(0, len(subject)))),
                   'reinst': conv_z2r(np.concatenate((np.nanmean(reinst_shift1,0), np.nanmean(reinst_shift0,0))))})
mu = [conv_z2r(np.nanmean(np.nanmean(reinst_shift0,0), 0)), conv_z2r(np.nanmean(np.nanmean(reinst_shift1,0), 0))]

plt.figure(figsize=(3.2,5))
ax = sns.violinplot(x="shift", y="reinst", data=df, inner=None, color='gray', split=False)
for i, mean_val in enumerate(mu):
    ax.plot([i -0.2, i +0.2], [mean_val, mean_val], color="black")  # Short black line
for i in range(len(subject)):
    ax.plot([0, 1], [conv_z2r(np.nanmean(reinst_shift0,0)[i]), conv_z2r(np.nanmean(reinst_shift1,0)[i])], color="black", linewidth=0.7, alpha=0.5)
    ax.scatter([0, 1], [conv_z2r(np.nanmean(reinst_shift0,0)[i]), conv_z2r(np.nanmean(reinst_shift1,0)[i])], color='black', s=5)
plt.show()
ax.spines[['right', 'top', 'bottom']].set_visible(False)
ax.set_xticklabels([]), ax.set_xticks([])
ax.set_yticklabels([]), ax.set_yticks([-0.04, 0, 0.04, 0.08])
plt.ylim([-0.064,0.084])
ax.tick_params(axis='both', which='major', length=6)

# neural reinstatement & behavioral retrieval
df = pd.DataFrame({'retr': np.concatenate((np.repeat(1, len(subject)), np.repeat(0, len(subject)))),
                   'reinst': conv_z2r(np.concatenate((np.nanmean(reinst_retr1,0), np.nanmean(reinst_retr0,0))))})
mu = [conv_z2r(np.nanmean(np.nanmean(reinst_retr0,0), 0)), conv_z2r(np.nanmean(np.nanmean(reinst_retr1,0), 0))]

plt.figure(figsize=(3.2,5))
ax = sns.violinplot(x="retr", y="reinst", data=df, inner=None, color='gray', split=False)
for i, mean_val in enumerate(mu):
    ax.plot([i -0.2, i +0.2], [mean_val, mean_val], color="black")  # Short black line
for i in range(len(subject)):
    ax.plot([0, 1], [conv_z2r(np.nanmean(reinst_retr0,0)[i]), conv_z2r(np.nanmean(reinst_retr1,0)[i])], color="black", linewidth=0.7, alpha=0.5)
    ax.scatter([0, 1], [conv_z2r(np.nanmean(reinst_retr0,0)[i]), conv_z2r(np.nanmean(reinst_retr1,0)[i])], color='black', s=5)
plt.show()
ax.spines[['right', 'top', 'bottom']].set_visible(False)
ax.set_xticklabels([]), ax.set_xticks([])
ax.set_yticklabels([]), ax.set_yticks([-0.04, 0, 0.04, 0.08])
plt.ylim([-0.064,0.084])
ax.tick_params(axis='both', which='major', length=6)