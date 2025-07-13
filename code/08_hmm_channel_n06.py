# ===========================================================================================
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# ----------------------------------
# 08 hmm_channel_n06.py
# ----------------------------------
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# ----------------------------------
# July, 2025
# ============================================================================================

from brainiak.eventseg.event import EventSegment
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import csv

##### INITIAL SETTINGS #####
label_encoder = LabelEncoder()

# Fit HMM Models and calculate match rates
def calculate_match_rate(eeg_channel_data, behavior_data_encoded, num_states):
    hmm = EventSegment(num_states)
    hmm.fit(eeg_channel_data)  

    # Identify neural transition points
    neural_transitions = np.diff(hmm.predict(eeg_channel_data), axis=0).nonzero()[0]

    # Identify behavioral transition points
    behavioral_transitions = np.diff(behavior_data_encoded, axis=0).nonzero()[0]

    # Calculate match rate with a window of ±3 sec
    window = 6
    match_count = 0

    for beh_transition in behavioral_transitions:
        if any((neural_transitions >= beh_transition - window) & (neural_transitions <= beh_transition + window)):
            match_count += 1

    match_rate = match_count / len(behavioral_transitions)
    return match_rate

sub_list = ['sub-01', 'sub-02', 'sub-19', 'sub-25', 'sub-30', 'sub-43']
clip_list = ['SP', 'CI', 'MO']

data_dir = "../dataset/eeg/pped/"

eeg_chan_list = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'FC6',
                 'F8', 'C4', 'F4', 'AF4', 'Fz', 'C3', 'F3', 'AF3', 'FC5', 'F7']
band_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']

feature_list = []

for chan in eeg_chan_list:
    for band in band_list:
        current_feature = band + "_" + chan
        feature_list.append(current_feature)

eeg_dict = {chan: list(range(i * 5, (i + 1) * 5)) for i, chan in enumerate(eeg_chan_list)}

##### MAIN ANALYSIS #####
hmm_match_cols = ['subjectkey', 'clip', 'feature', 'match_rate'] + [f'perm_{str(i).zfill(4)}' for i in range(1, 1001)]
hmm_match_df = pd.DataFrame(columns = hmm_match_cols)
iter_idx = 1

total_iter = len(sub_list) * len(clip_list) * len(eeg_dict)

for sub in sub_list:
    for clip in clip_list:
        
        ### Data preparation
        tmp_filename = sub + "_" + clip
        tmp_eeg_df = pd.read_csv(data_dir + sub + "/CEBRA_input/" + tmp_filename + "-STFT-CEBRA.csv")
        tmp_eeg_np = tmp_eeg_df.iloc[:, :-1].values  # exclude last column (valence)
        tmp_val = tmp_eeg_df.iloc[:, -1].values      # only last column (valence)
        tmp_val_encoded = label_encoder.fit_transform(tmp_val)
        tmp_n_states = len(np.diff(tmp_val_encoded, axis = 0).nonzero()[0]) + 1
        
        ### Calculation of match rate for every feature
        for chan_idx in range(len(eeg_dict)):
            tmp_chan = list(eeg_dict.keys())[chan_idx]
            tmp_feat_idx = eeg_dict[tmp_chan]
            tmp_eeg_input = tmp_eeg_np[:, tmp_feat_idx]

            ## original match rate
            feat_match_rate = calculate_match_rate(tmp_eeg_input, tmp_val_encoded, tmp_n_states)
            appended_row = np.array([sub, clip, tmp_chan, feat_match_rate])

            ## permutation test
            for perm_idx in range(1000):
                perm_idx = perm_idx + 1

                # shuffle the EEG time-series
                np.random.seed(perm_idx) # fixing the seed
                tmp_perm_idx = np.random.permutation(tmp_eeg_input.shape[0]) # randomly shuffle the row index
                tmp_perm_input = tmp_eeg_input[tmp_perm_idx]

                # calculate the permuted match rate
                tmp_perm_match_rate = calculate_match_rate(tmp_perm_input, tmp_val_encoded, tmp_n_states)

                appended_row = np.append(appended_row, tmp_perm_match_rate)
                print(f">>>>> [{perm_idx:04d}/1000] permutation done!") 
            
            new_appended_row = pd.DataFrame([appended_row], columns = hmm_match_cols)
            hmm_match_df = pd.concat([hmm_match_df, new_appended_row], ignore_index = True)
                        
            print(f"[{iter_idx:04d}/{total_iter}] HMM analysis done! (channel) {tmp_chan} / (sub) {sub} / (clip) {clip}") 
            print("=====================================================================================================")

            iter_idx = iter_idx + 1
            
out_fname = "../results/hmm_match_rate/hmm_match_rate_channel.csv"
hmm_match_df.to_csv(out_fname, index=False)
print(f"Results saved to {out_fname}")
