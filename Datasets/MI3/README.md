# MI3 Motor-Imagery Dataset Notes

The MI3 dataset contains 25 right-handed participants performing three conditions (right-hand motor imagery, right-elbow motor imagery, rest with eyes open) across two sessions. Each recording lasts 8 s (4 s cue + 4 s imagery) resulting in $25\text{ subjects} \times 3\text{ classes} \times 300\text{ trials} = 22{,}500$ trials. Acquisition details and experimental protocol are documented in the original publication by Xuelin Ma et al. (Scientific Data, 2020) and the public release at <https://doi.org/10.7910/DVN/RBN3XG>.

## Preprocessing requirements

All scripts rely on EEGLAB plus several add-ons that must be installed manually in MATLAB and added to the path before executing any `.m` file:

| Toolkit | Purpose | Download / documentation |
| --- | --- | --- |
| EEGLAB | Core EEG processing toolbox, launches `eeglab` GUI/nogui session | <https://sccn.ucsd.edu/eeglab/download.php> |
| Neuroscan CNT import plugin (`neuroscanio`) | Reads the `.cnt` files inside `sourcedata/` | <https://github.com/sccn/eeglab/tree/develop/plugins/neuroscanio> |
| EEGLAB BIDS tools | Writes BIDS-compliant metadata (`task-motorimagery_*.json/.tsv`) alongside `.set` files | <https://github.com/sccn/eeglab/tree/develop/plugins/bids-matlab-tools> |
| BioSig | Provides additional data readers and filters referenced by EEGLAB | <https://biosig.sourceforge.net/> |
| AAR (Automatic Artifact Removal) plugin | Supplies `pop_autobsseog` and related ICA/SOBI helpers for ocular/EMG cleanup | <https://github.com/cincibrainlab/AAR> |

After downloading, unzip each extension inside your EEGLAB `plugins/` folder or install them via **File ▸ Manage extensions** in the EEGLAB GUI. Update the `eeglab_path` variable near the top of every script so MATLAB can locate your installation.

## MATLAB preprocessing script

The current workflow is handled entirely by `code/MI3_process.m`, which follows these stages:

1. Enumerate subjects (`ListSub`) and sessions (EEG recordings) inside `sourcedata/`.
2. Load raw CNT files with `pop_loadcnt` and enrich channel metadata via `pop_chanedit(..., 'lookup', 'code/channel_dict.ced')`.
3. Strip non-EEG channels, apply a $2$–$45\,$Hz band-pass, re-reference with common average, and remove ocular/EMG components using `pop_autobsseog`.
4. Segment 4 s trials for motor imagery vs rest, build label vectors (0 rest / 1 elbow / 2 hand), and concatenate across sessions.
5. Persist cleaned EEGLAB objects and aggregated tensors to the `derivatives/` folder for downstream PyTorch training.

### `MI3_process.m`

- **Scope:** Batch-processes every subject/session from `Datasets\MI3\sourcedata` and exports aggregated `.mat` tensors (all trials/labels) to `Datasets\MI3\derivatives`.
- **Key parameters:**
	- `NumSub`, `NumSes`, `NumTech` control which subject/session/modality indices are processed (default starts at index 3 to skip `.`/`..`). *IT CAN BE USED TO LIMIT THE PROCESSING TO ONLY THE FIRST SUBJECT BY SLICING THE NUMSUB to [3:4]*
	- `pop_eegfiltnew(... 'locutoff', 2, 'hicutoff', 45)` implements the $2$–$45\,$Hz band-pass prior to artifact removal.
	- Trials are kept at the native sampling rate, then optionally resampled later when writing `all_data` tensors (`trial_duration = 4` s, labels: 0 rest / 1 elbow / 2 hand).
	- Saves `subject_file = saveDir/[Subject]_eeg.mat` containing `all_data` (N×C×T) and `all_label` vectors, with class histograms printed at the end.
- **When to use:** MI3 dataset ingestion before training convolutional networks.


## Folder structure

```text
MI3/
├── dataset_description.json      # BIDS descriptor for the study
├── participants.json/.tsv        # Subject demographics and session info
├── README.md                    # This file describing the dataset and processing scripts
├── task-motorimagery_channels.tsv
├── task-motorimagery_coordsystem.json
├── task-motorimagery_eeg.json
├── task-motorimagery_electrodes.tsv
├── task-motorimagery_events.json
├── code/
│   ├── channel_dict.ced          # Standard 64-ch montage
│   ├── Customchannel_dict.ced    # Alternate montage with dropped sensors
│   └── MI3_process.m             # Batch MI3 preprocessing/export
├── derivatives/
│   ├── Readme.MD                 # Notes about derivative files
│   ├── sub-011_eeg.mat           # All-session tensor (4 s epochs)
│   └── sub-011_eeg90hz.mat       # Down-sampled derivative
└── sourcedata/
	└── sub-011/
		├── ses-01/eeg/sub-011_ses-01_task-motorimagery_eeg.cnt
		└── ses-02/eeg/sub-011_ses-02_task-motorimagery_eeg.cnt
```

- `dataset_description.json`, `participants.*`, and `task-motorimagery_*` files supply the metadata required by the BIDS spec (channel names, coordinates, reference electrode, amplifier settings, event codes).
- The `code/` directory now contains the MI3 batch preprocessing script plus the channel-location dictionaries consumed by EEGLAB.
- `sourcedata/` includes the raw CNT acquisitions captured for subject `sub-011` across `ses-01` and `ses-02` (EEG only in this workspace snapshot).
- `derivatives/` stores the consolidated `.mat` tensors (`sub-011_eeg.mat` and its 90 Hz down-sampled variant) consumed by the PyTorch notebook.




## Additional notes

- The `.ced` files under `code/` must stay synchronized with the physical cap used in MI3; edit `channel_dict.ced` only if you re-label sensors inside EEGLAB.
- After running any script, inspect the generated `.set` files with `pop_eegplot` to confirm ocular artifact removal worked as expected before launching the PyTorch training notebook.