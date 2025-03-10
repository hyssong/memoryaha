# memory retrieval, causal relationship, and 6 narrative features
# Mar 8, 2025, Hayoung Song

import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.anova import anova_lm

def conv_r2z(r):
    eps = 1e-10  # Small value to prevent division by zero
    r[np.where(r==1)] = 1-eps
    r[np.where(r==-1)] = -1+eps
    with np.errstate(invalid='ignore', divide='ignore'):
        return 0.5 * (np.log(1 + r) - np.log(1 - r))
def conv_z2r(z):
    with np.errstate(invalid='ignore', divide='ignore'):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

directory = '/folder'
memory_retrieval = np.array(pd.read_csv(directory+'/data/memory_retrieval.csv', header=None))
causal_relationship = np.array(pd.read_csv(directory+'/data/causal_relationship.csv', header=None))
semantic_similarity = conv_r2z(np.array(pd.read_csv(directory+'/data/narrative_feature_semantic_similarity_r.csv', header=None)))
character_similarity = conv_r2z(np.array(pd.read_csv(directory+'/data/narrative_feature_character_similarity_r.csv', header=None)))
place_similarity = conv_r2z(np.array(pd.read_csv(directory+'/data/narrative_feature_place_similarity_r.csv', header=None)))
visual_similarity = conv_r2z(np.array(pd.read_csv(directory+'/data/narrative_feature_visual_similarity_r.csv', header=None)))
audio_similarity = conv_r2z(np.array(pd.read_csv(directory+'/data/narrative_feature_audio_similarity_r.csv', header=None)))
time_proximity = np.array(pd.read_csv(directory+'/data/narrative_feature_time_proximity.csv', header=None))

# relationship between memory retrieval & causal relationship
idd = np.where(~np.isnan(causal_relationship))
scipy.stats.spearmanr(memory_retrieval[idd], causal_relationship[idd], nan_policy='omit')

# visualize pairwise relationship
df = pd.DataFrame(np.concatenate((scipy.stats.zscore(memory_retrieval[idd].reshape(-1,1), nan_policy='omit'),
                                  scipy.stats.zscore(causal_relationship[idd].reshape(-1,1)),
                                  scipy.stats.zscore(semantic_similarity[idd].reshape(-1,1)),
                                  scipy.stats.zscore(character_similarity[idd].reshape(-1,1)),
                                  scipy.stats.zscore(place_similarity[idd].reshape(-1,1)),
                                  scipy.stats.zscore(time_proximity[idd].reshape(-1,1)),
                                  scipy.stats.zscore(visual_similarity[idd].reshape(-1,1)),
                                  scipy.stats.zscore(audio_similarity[idd].reshape(-1,1))),1),
                  columns=['memory', 'causal', 'semantic', 'character', 'place', 'time', 'visual', 'audio'])
d = scipy.stats.spearmanr(np.array(df), nan_policy='omit')[0]
d[np.where(np.triu(np.zeros((8,8))+1,1)==0)] = np.nan
plt.figure(), sns.heatmap(d, vmin=0, vmax=0.87, annot=True)
plt.xticks([]), plt.yticks([])

# explained variance
df = df.dropna()

y = df['memory']
X = df[['causal','semantic', 'visual', 'audio', 'character', 'place', 'time']]
X = sm.add_constant(X)
model_wcausal = sm.OLS(y, X).fit()
model_wcausal.summary()

y = df['memory']
X = df[['semantic', 'visual', 'audio', 'character', 'place', 'time']]
X = sm.add_constant(X)
model_wocausal = sm.OLS(y, X).fit()
model_wocausal.summary()

y = df['memory']
X = df[['causal']]
X = sm.add_constant(X)
model_onlycausal = sm.OLS(y, X).fit()
model_onlycausal.summary()

anova_results = anova_lm(model_wocausal, model_wcausal)
print(anova_results)

anova_results = anova_lm(model_onlycausal, model_wcausal)
print(anova_results)
