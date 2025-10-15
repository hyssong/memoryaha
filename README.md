# memoryaha

Preprint can be found [here](https://www.biorxiv.org/content/10.1101/2025.03.12.642853v1).
Raw and processed fMRI data are shared in [OpenNeuro](https://openneuro.org/datasets/ds005658).

## installation
To run the codes, which reproduces results in the manuscript, it is necessary to install the Python packages included in environment.yml. We recommend installing conda and executing the following commands. This takes less than a minute in a standard laptop.
```bash
conda env create -f environment.yml
conda activate memoryaha
```
R v4.4.2 was used to run the R code.

## code
- **code_extractbold.py**: extracts BOLD activity time series of voxels within the chosen parcels
- **code_ahasynchrony.py**: process aha button press data (ahabutton.csv) and calculate button press synchrony (Figure 1D)
- **code_causalitymemory.py**: load memory retrieval matrix, causal relationship matrix, and various narrative feature similarity matrices and analyze the relationships between them (Figure 2, Figure S1)
- **code_hmm.py**: conduct hidden Markov model analysis to calculate neural pattern shifts (Figure 3, Figure S2)
- **code_ahahmmreinst.py**: categorize aha button presses based on neural pattern shifts and behavioral retrieval and align neural reinstatement at aha button press moments (Figure 4)
- **code_ahahmmreinst_stats.R**: conduct logistic generalized linear mixed effect models

## data
**sceneindex**<br />
48 segmented events
- scene: This is Us Season 1 Episode 1 segmented into 48 events.
- start: Start time of the event. The time index was recorded from Adobe Premiere Pro. To prevent bleeding of audio from one event to the next, manual movie edits were conducted at the start or end of some of the events. The .mp4 files used in the experiment are not shared.
- duration: Duration of the event.
- nTR: Number of TRs the event was played during the fMRI experiment.
- importance: Rated importance of the event, averaged across 8 raters.
- event: Main character(s) of the event.
- description: Short event description by the first author.
- binary indices of which characters appeared and where the event took place.

**groupscene**<br />
Three scrambled-order groups
- run: Indices of 10 fMRI runs.
- order: Event order within a run. 3 events for run 7 and 5 events for the rest of the runs.
- sceneid: Scene index corresponds to sceneindex.csv.
- blockid: Scene order within a block was fixed, and block order differed across scrambled-order groups.
- char: (1) Jack, (2) Kate, (3) Randall, (4) Kevin, (5) Kevin & Kate.

**ahabutton**<br />
Moments of aha button presses, in TR resolution
Analysis of the aha button press data is demonstrated in: /code/code_ahasynchrony.py<br>
- subject: 36 fMRI participant index. (Some of the subject numbers are skipped not because they were excluded from analyses, but because either the participant did not show up to the experiment or did not complete the entire session.)
- run: At which fMRI run the button was pressed.
- scene: Corresponds to "order" in groupscene.csv.
- TR (scene): Moments of aha button press (in TR) within the scene duration.
- TR (run): Moments of aha button press (in TR) within the fMRI run duration.

**causal_relationship, memory_retrieval, narrative_feature_*** <br />
Event-by-event (48 x 48 matrix) matrix, with events in original order (scene index from 1 to 48)
