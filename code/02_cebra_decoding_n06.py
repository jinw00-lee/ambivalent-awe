# ============================================================================================
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# ----------------------------------
# 02 cebra_decoding_n06.py
# ----------------------------------
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# ----------------------------------
# July, 2025
# ============================================================================================

### IMPORT MODULES
from sklearn.cross_decomposition import CCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np
import random
import cebra
import csv
import sys
import os


### INITIAL SETTINGS 
sub_list_df = pd.read_csv("../dataset/sub-list-decoding_n06.csv", header = None)
sub_list = np.array(sub_list_df[0])

clip_list = ['SP', 'CI', 'MO']
dim_list = [1, 2, 3, 4, 5, 6, 7, 8, 9]

data_dir = '../dataset/eeg/pped/'
eeg_chan_list = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'FC6',
                 'F8', 'C4', 'F4', 'AF4', 'Fz', 'C3', 'F3', 'AF3', 'FC5', 'F7']
band_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']

feature_list = []

for chan in eeg_chan_list:
    for band in band_list:
        current_feature = band + "_" + chan
        feature_list.append(current_feature)
        

### TRAINING A CEBRA MODEL FOR EACH SUBJECT'S CLIP DATA
"""
<NOTE!>
@ CEBRA produces consistent embeddings, but it currently does not support a feature to fix the seed.
@ To replicate our results, please use our pre-trained models.
@ We designed the code to load the saved pretrained model and use it for decoding analysis. 
@ Therefore, please execute the code while keeping the footnotes for this part intact.

iter_idx = 0
total_idx = len(sub_list) * len(clip_list) * len(dim_list)

for target_sub in sub_list:
    for clip in clip_list:
        
        # data loading
        tmp_csv_name = target_sub + "_" + clip + "-STFT-CEBRA.csv"
        tmp_csv = pd.read_csv(os.path.join(data_dir, target_sub, 'CEBRA_input', tmp_csv_name))
        
        tmp_eeg = tmp_csv.iloc[:, :-1].to_numpy()
        tmp_val = tmp_csv.iloc[:, -1].to_numpy()
        
        for dim in dim_list:
            iter_idx += 1
            
            # model training
            tmp_cebra_model = cebra.CEBRA(batch_size = len(tmp_eeg),                  
                                          model_architecture = 'offset10-model',   
                                          num_hidden_units = 38,                   
                                          learning_rate = 1e-3,                 
                                          output_dimension = dim, 
                                          max_iterations = 500,
                                          temperature_mode = 'auto',
                                          device = 'cuda',
                                          hybrid = False,
                                          verbose = True)
        
            tmp_cebra_model.fit(tmp_eeg, tmp_val)
            
            # save the model
            model_name = "../results/cebra_models/dim-" + str(dim) + "/CEBRA_d-" + str(dim) + "_" + target_sub + "_" + clip + ".pt" 
            tmp_cebra_model.save(model_name)
            
            print(f"[{iter_idx:03d}/{total_idx}] CEBRA model was successfully saved for {target_sub}'s {clip} data! (dim = {dim})")
"""


### DECODING ANALYSIS (1): ACROSS-PARTICIPANTS
# result template
across_sub_pred_df = pd.DataFrame(columns = ['ref_sub', 'target_sub', 'clip', 'dim', 
                                             'test_acc_random', 'test_f1_random',
                                             'test_acc_not_align', 'test_f1_not_align',
                                             'test_acc_align', 'test_f1_align'])
for dim in dim_list:
    for ref_sub in sub_list:
        target_list = [x for x in sub_list if x != ref_sub]
    
        for target_sub in target_list:
            for clip in clip_list:
                
                
                ## PART A. EMBEDDING EXTRACTION ===================================================
                # A-1. embedding setup for reference participant
                ref_model_name = "../results/cebra_models/dim-" + str(dim) + "/CEBRA_d-" + str(dim) + "_" + ref_sub + "_" + clip + ".pt"
                ref_model = cebra.CEBRA.load(ref_model_name)
                
                ref_csv_name = ref_sub + "_" + clip + "-STFT-CEBRA.csv"
                ref_csv = pd.read_csv(os.path.join(data_dir, ref_sub, 'CEBRA_input', ref_csv_name))
                ref_eeg = ref_csv.iloc[:, :-1].to_numpy()
                ref_val = ref_csv.iloc[:, -1].to_numpy()
                
                ref_emb = ref_model.transform(ref_eeg)
                
                # A-2. embedding setup for target participant
                target_model_name = "../results/cebra_models/dim-" + str(dim) + "/CEBRA_d-" + str(dim) + "_" + target_sub + "_" + clip + ".pt"
                target_model = cebra.CEBRA.load(target_model_name)
                
                target_csv_name = target_sub + "_" + clip + "-STFT-CEBRA.csv"
                target_csv = pd.read_csv(os.path.join(data_dir, target_sub, 'CEBRA_input', target_csv_name))
                target_eeg = target_csv.iloc[:, :-1].to_numpy()
                target_val = target_csv.iloc[:, -1].to_numpy()
                
                target_emb = target_model.transform(target_eeg)
        
                # A-3. embedding alignment 
                cca = CCA(n_components = dim)
                cca.fit(ref_emb, target_emb)
            
                ref_emb_align, target_emb_align = cca.transform(ref_emb, target_emb)
                
                
                ## PART B. DECODING ANALYSIS =====================================================
                # B-1. random condition
                knn_random = KNeighborsClassifier(n_neighbors = 15, metric = 'cosine')
                np.random.seed(1234)
                knn_random.fit(ref_emb, np.random.permutation(ref_val))
                target_pred_random = knn_random.predict(target_emb)
            
                test_acc_random = accuracy_score(target_val, target_pred_random)
                test_f1_random = f1_score(target_val, target_pred_random, average = 'weighted')
            
                # B-2. not-aligned condition
                knn_not_align = KNeighborsClassifier(n_neighbors = 15, metric = 'cosine')
                knn_not_align.fit(ref_emb, ref_val)
                target_pred_not_align = knn_not_align.predict(target_emb)
            
                test_acc_not_align = accuracy_score(target_val, target_pred_not_align)
                test_f1_not_align = f1_score(target_val, target_pred_not_align, average = 'weighted')
            
                # B-3. aligned condition
                knn_align = KNeighborsClassifier(n_neighbors = 15, metric = 'cosine')
                knn_align.fit(ref_emb_align, ref_val)
                target_pred_align = knn_align.predict(target_emb_align)
            
                test_acc_align = accuracy_score(target_val, target_pred_align)
                test_f1_align = f1_score(target_val, target_pred_align, average = 'weighted')
            
                
                ### PART C. SAVE THE RESULTS =====================================================
                appended_row = pd.DataFrame([{'ref_sub': ref_sub,
                                              'target_sub': target_sub,
                                              'clip': clip,
                                              'dim': dim, 
                                              'test_acc_random': test_acc_random,
                                              'test_f1_random': test_f1_random,
                                              'test_acc_not_align': test_acc_not_align,
                                              'test_f1_not_align': test_f1_not_align,
                                              'test_acc_align': test_acc_align,
                                              'test_f1_align': test_f1_align}])
            
                across_sub_pred_df = pd.concat([across_sub_pred_df, appended_row])
                print(f"Pairwise decoding was done!: [CLIP] {clip} | [REF] {ref_sub} | [TARGET] {target_sub} | [DIM] {str(dim)}")
                
across_sub_pred_df.to_csv("../results/decoding_performances/across_sub_decoding_CEBRA.csv", index = False)


### DECODING ANALYSIS (2): ACROSS-CLIPS
# result template
across_clip_pred_df = pd.DataFrame(columns = ['ref_clip', 'target_clip', 'sub', 'dim', 
                                              'test_acc_random', 'test_f1_random',
                                              'test_acc_not_align', 'test_f1_not_align',
                                              'test_acc_align', 'test_f1_align'])

for dim in dim_list:
    for ref_clip in clip_list:
        target_list = [x for x in clip_list if x != ref_clip]
    
        for target_clip in target_list:
            for sub in sub_list:
                
                ## PART A. EMBEDDING EXTRACTION ===================================================
                # A-1. embedding setup for reference participant
                ref_model_name = "../results/cebra_models/dim-" + str(dim) + "/CEBRA_d-" + str(dim) + "_" + sub + "_" + ref_clip + ".pt"
                ref_model = cebra.CEBRA.load(ref_model_name)
                
                ref_csv_name = sub + "_" + ref_clip + "-STFT-CEBRA.csv"
                ref_csv = pd.read_csv(os.path.join(data_dir, sub, 'CEBRA_input', ref_csv_name))
                ref_eeg = ref_csv.iloc[:, :-1].to_numpy()
                ref_val = ref_csv.iloc[:, -1].to_numpy()
                
                ref_emb = ref_model.transform(ref_eeg)
                
                # A-2. embedding setup for target participant
                target_model_name = "../results/cebra_models/dim-" + str(dim) + "/CEBRA_d-" + str(dim) + "_" + sub + "_" + target_clip + ".pt"
                target_model = cebra.CEBRA.load(target_model_name)
                
                target_csv_name = sub + "_" + target_clip + "-STFT-CEBRA.csv"
                target_csv = pd.read_csv(os.path.join(data_dir, sub, 'CEBRA_input', target_csv_name))
                target_eeg = target_csv.iloc[:, :-1].to_numpy()
                target_val = target_csv.iloc[:, -1].to_numpy()
                
                target_emb = target_model.transform(target_eeg)
                
                # A-3. embedding alignment 
                cca = CCA(n_components = dim)
                cca.fit(ref_emb, target_emb)
            
                ref_emb_align, target_emb_align = cca.transform(ref_emb, target_emb)
                
                
                ## PART B. DECODING ANALYSIS =====================================================
                # B-1. random condition
                knn_random = KNeighborsClassifier(n_neighbors = 15, metric = 'cosine')
                np.random.seed(1234)
                knn_random.fit(ref_emb, np.random.permutation(ref_val))
                target_pred_random = knn_random.predict(target_emb)
            
                test_acc_random = accuracy_score(target_val, target_pred_random)
                test_f1_random = f1_score(target_val, target_pred_random, average = 'weighted')
            
                # B-2. not-aligned condition
                knn_not_align = KNeighborsClassifier(n_neighbors = 15, metric = 'cosine')
                knn_not_align.fit(ref_emb, ref_val)
                target_pred_not_align = knn_not_align.predict(target_emb)
            
                test_acc_not_align = accuracy_score(target_val, target_pred_not_align)
                test_f1_not_align = f1_score(target_val, target_pred_not_align, average = 'weighted')
            
                # B-3. aligned condition
                knn_align = KNeighborsClassifier(n_neighbors = 15, metric = 'cosine')
                knn_align.fit(ref_emb_align, ref_val)
                target_pred_align = knn_align.predict(target_emb_align)
            
                test_acc_align = accuracy_score(target_val, target_pred_align)
                test_f1_align = f1_score(target_val, target_pred_align, average = 'weighted')
                
                
                ### PART C. SAVE THE RESULTS =====================================================
                appended_row = pd.DataFrame([{'ref_clip': ref_clip,
                                              'target_clip': target_clip,
                                              'sub': sub,
                                              'dim': dim, 
                                              'test_acc_random': test_acc_random,
                                              'test_f1_random': test_f1_random,
                                              'test_acc_not_align': test_acc_not_align,
                                              'test_f1_not_align': test_f1_not_align,
                                              'test_acc_align': test_acc_align,
                                              'test_f1_align': test_f1_align}])
            
                across_clip_pred_df = pd.concat([across_clip_pred_df, appended_row])
                print(f"Pairwise decoding was done!: [SUB] {sub} | [REF] {ref_clip} | [TARGET] {target_clip} | [DIM] {str(dim)}")
                
across_clip_pred_df.to_csv("../results/decoding_performances/across_clip_decoding_CEBRA.csv", index = False)
