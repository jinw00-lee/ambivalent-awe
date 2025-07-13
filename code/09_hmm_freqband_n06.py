# ===========================================================================================
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# ----------------------------------
# 09 hmm_freqband_n06.py
# ----------------------------------
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# ----------------------------------
# May, 2025
# ============================================================================================

from brainiak.eventseg.event import EventSegment
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv

##### INITIAL SETTINGS #####
label_encoder = LabelEncoder()

# Fit HMM Models and calculate match rates
def calculate_match_rate(eeg_band_data, behavior_data_encoded, num_states):
    hmm = EventSegment(num_states)
    hmm.fit(eeg_band_data)

    # Identify neural transition points
    neural_states = hmm.predict(eeg_band_data)
    neural_transitions = np.diff(neural_states, axis=0).nonzero()[0]

    # Identify behavioral transition points
    behavioral_transitions = np.diff(behavior_data_encoded, axis=0).nonzero()[0]

    # Calculate match rate with a window of ±3 sec (assuming 1 sample = 1 sec)
    window = 6
    match_count = 0
    for beh_transition in behavioral_transitions:
        if np.any((neural_transitions >= beh_transition - window) &
                  (neural_transitions <= beh_transition + window)):
            match_count += 1
    match_rate = match_count / len(behavioral_transitions)
    return match_rate

# Subjects and clips
sub_list = ['sub-01', 'sub-02', 'sub-19', 'sub-25', 'sub-30', 'sub-43']
clip_list = ['SP', 'CI', 'MO']

# Directory and band definitions
data_dir = "../dataset/eeg/pped/"
band_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']

eeg_chan_list = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'FC6',
                 'F8', 'C4', 'F4', 'AF4', 'Fz', 'C3', 'F3', 'AF3', 'FC5', 'F7']

band_dict = {band: [] for band in band_list}

for chan_idx, chan in enumerate(eeg_chan_list):
    for band_idx, band in enumerate(band_list):
        
        # index in flat feature array
        feature_idx = chan_idx * len(band_list) + band_idx
        band_dict[band].append(feature_idx)

# Prepare DataFrame to collect results
hmm_match_cols = ['subjectkey', 'clip', 'band', 'match_rate'] + [f'perm_{i:04d}' for i in range(1, 1001)]
hmm_match_df = pd.DataFrame(columns=hmm_match_cols)
iter_idx = 1
total_iter = len(sub_list) * len(clip_list) * len(band_list)

###### MAIN ANALYSIS #####
for sub in sub_list:
    for clip in clip_list:

        # Load data
        fname = f"{data_dir}{sub}/CEBRA_input/{sub}_{clip}-STFT-CEBRA.csv"
        tmp_df = pd.read_csv(fname)
        eeg_np = tmp_df.iloc[:, :-1].values  # all band features
        valence = tmp_df.iloc[:, -1].values  # behavior labels
        val_enc = label_encoder.fit_transform(valence)

        # Determine number of states from behavioral transitions
        num_states = np.diff(val_enc, axis=0).nonzero()[0].size + 1

        for band in band_list:

            # Extract multi-channel band data
            idxs = band_dict[band]
            eeg_band_data = eeg_np[:, idxs]

            # Compute original match rate
            base_rate = calculate_match_rate(eeg_band_data, val_enc, num_states)
            row = [sub, clip, band, base_rate]

            # Permutation test
            for p in range(1, 1001):
                np.random.seed(p)
                perm_idx = np.random.permutation(eeg_band_data.shape[0])
                perm_data = eeg_band_data[perm_idx]
                rate_p = calculate_match_rate(perm_data, val_enc, num_states)
                row.append(rate_p)
                print(f"Permutation {p:04d}/1000 for band {band} done.")

            # Append to DataFrame
            df_row = pd.DataFrame([row], columns=hmm_match_cols)
            hmm_match_df = pd.concat([hmm_match_df, df_row], ignore_index=True)

            print(f"[{iter_idx:04d}/{total_iter}] Completed band {band} for {sub} clip {clip}.")
            iter_idx += 1

# Save results
from brainiak.eventseg.event import EventSegment
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np

##### INITIAL SETTINGS #####
label_encoder = LabelEncoder()

# Fit HMM Models and calculate match rates
def calculate_match_rate(eeg_band_data, behavior_data_encoded, num_states):
    """
    eeg_band_data: array of shape (timepoints, num_channels)
    behavior_data_encoded: 1D array of encoded behavioral states
    num_states: int, number of HMM states to fit
    """
    hmm = EventSegment(num_states)
    hmm.fit(eeg_band_data)

    # Identify neural transition points
    neural_states = hmm.predict(eeg_band_data)
    neural_transitions = np.diff(neural_states, axis=0).nonzero()[0]

    # Identify behavioral transition points
    behavioral_transitions = np.diff(behavior_data_encoded, axis=0).nonzero()[0]

    # Calculate match rate with a window of ±3 sec (assuming 1 sample = 1 sec)
    window = 6
    match_count = 0
    for beh_transition in behavioral_transitions:
        if np.any((neural_transitions >= beh_transition - window) &
                  (neural_transitions <= beh_transition + window)):
            match_count += 1
    match_rate = match_count / len(behavioral_transitions)
    return match_rate

# Subjects and clips
sub_list = ['sub-01', 'sub-02', 'sub-19', 'sub-25', 'sub-30', 'sub-43']
clip_list = ['SP', 'CI', 'MO']

# Directory and band definitions
data_dir = "../dataset/eeg/pped/"
band_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']

# Build index mapping: band -> list of feature indices (assuming CSV columns order matches STFT bands per channel)
eeg_chan_list = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'FC6',
                 'F8', 'C4', 'F4', 'AF4', 'Fz', 'C3', 'F3', 'AF3', 'FC5', 'F7']
# Each channel has 5 band columns in order: delta, theta, alpha, beta, gamma
# So overall CSV columns before valence: [delta_P7, theta_P7, ..., gamma_P7, delta_P4, ..., gamma_F7]
band_dict = {band: [] for band in band_list}
for chan_idx, chan in enumerate(eeg_chan_list):
    for band_idx, band in enumerate(band_list):
        # index in flat feature array
        feature_idx = chan_idx * len(band_list) + band_idx
        band_dict[band].append(feature_idx)

# Prepare DataFrame to collect results
hmm_match_cols = ['subjectkey', 'clip', 'band', 'match_rate'] + [f'perm_{i:04d}' for i in range(1, 1001)]
hmm_match_df = pd.DataFrame(columns=hmm_match_cols)
iter_idx = 1
total_iter = len(sub_list) * len(clip_list) * len(band_list)

# Main analysis: loop over subjects, clips, and bands
for sub in sub_list:
    for clip in clip_list:
        # Load data
        fname = f"{data_dir}{sub}/CEBRA_input/{sub}_{clip}-STFT-CEBRA.csv"
        tmp_df = pd.read_csv(fname)
        eeg_np = tmp_df.iloc[:, :-1].values  # all band features
        valence = tmp_df.iloc[:, -1].values  # behavior labels
        val_enc = label_encoder.fit_transform(valence)
        # Determine number of states from behavioral transitions
        num_states = np.diff(val_enc, axis=0).nonzero()[0].size + 1

        for band in band_list:
            # Extract multi-channel band data
            idxs = band_dict[band]
            eeg_band_data = eeg_np[:, idxs]

            # Compute original match rate
            base_rate = calculate_match_rate(eeg_band_data, val_enc, num_states)
            row = [sub, clip, band, base_rate]

            # Permutation test
            for p in range(1, 1001):
                np.random.seed(p)
                perm_idx = np.random.permutation(eeg_band_data.shape[0])
                perm_data = eeg_band_data[perm_idx]
                rate_p = calculate_match_rate(perm_data, val_enc, num_states)
                row.append(rate_p)
                print(f"Permutation {p:04d}/1000 for band {band} done.")

            # Append to DataFrame
            df_row = pd.DataFrame([row], columns=hmm_match_cols)
            hmm_match_df = pd.concat([hmm_match_df, df_row], ignore_index=True)

            print(f"[{iter_idx:04d}/{total_iter}] Completed band {band} for {sub} clip {clip}.")
            iter_idx += 1

# Save results
out_fname = "../results/hmm_match_rate/hmm_match_rate_freqband.csv"
hmm_match_df.to_csv(out_fname, index=False)
print(f"Results saved to {out_fname}")



