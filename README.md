# memoryaha

Preprint can be found [here](https://www.biorxiv.org/content/10.1101/2025.03.12.642853v1).
Raw and processed fMRI data are shared in [OpenNeuro](https://openneuro.org/datasets/ds005658).

## code
- **code_extractbold.py**: extracts BOLD activity time series of voxels within the chosen parcels
- **code_ahasynchrony.py**: process aha button press data (ahabutton.csv) and calculate button press synchrony (Figure 1D)
- **code_causalitymemory.py**: load memory retrieval matrix, causal relationship matrix, and various narrative feature similarity matrices and analyze the relationships between them (Figure 2, Figure S1)
- **code_hmm.py**: conduct hidden Markov model analysis to calculate neural pattern shifts (Figure 3, Figure S2)
- **code_ahahmmreinst.py**: categorize aha button presses based on neural pattern shifts and behavioral retrieval and align neural reinstatement at aha button press moments (Figure 4)
- **code_ahahmmreinst_stats.R**: conduct logistic generalized linear mixed effect models
