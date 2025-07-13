# ============================================================================================
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# ----------------------------------
# 06 CEBRA_LATENT_SPACE_CONSTRUCTION_n27.py
# ----------------------------------
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# ----------------------------------
# July, 2025
# ============================================================================================
"""
============
### NOTE
============
This code fits single CEBRA model and 1,000 permuted models for statistical testing per participant-clip pair. 
For efficient reproduction of results, we recommend proceeding directly to '11_latent_stat_analysis_n27.R' 
to reproduce the results described in Fig 5 using precomputed statistics based on the pretrained models. 
This is because 1) CEBRA currently does not support seed fixation, and
2) the model fitting process for all participants takes a considerable amount of time.

"""

### IMPORT MODULES
from utils.custom_func import medoid, sum_with_nan, avg_d_calculator, med_d_calculator
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from collections import defaultdict

import pandas as pd
import numpy as np
import random
import cebra
import csv
import sys
import os

### INITIAL SETTINGS
if len(sys.argv) > 1:
    target_sub = sys.argv[1]  # Get target_sub from command-line argument
else:
    print("Usage: python3 09_cebra_latent_space_construction.py <target_sub>")
    sys.exit(1)               # Exit the script if no target_sub is provided

clip_list = ['SP', 'CI', 'MO']
dim = 7
perm_test_num = 1000

data_dir = '../dataset/eeg/pped/'
eeg_chan_list = ['P7', 'P4', 'Cz', 'Pz', 'P3', 'P8', 'O1', 'O2', 'FC6',
                 'F8', 'C4', 'F4', 'AF4', 'Fz', 'C3', 'F3', 'AF3', 'FC5', 'F7']
band_list = ['delta', 'theta', 'alpha', 'beta', 'gamma']

feature_list = [] # to fill the feature names iteratively (eeg_chan_list * band_list)

for chan in eeg_chan_list:
    for band in band_list:
        current_feature = band + "_" + chan
        feature_list.append(current_feature)
        
# a subject-level result report template
target_sub_df = pd.DataFrame(columns = ['subjectkey', 'clip', 'dim',
                                        'real_silh_avg_total', 'real_silh_sd_total',
                                        'real_silh_avg_neu', 'real_silh_sd_neu',
                                        'real_silh_avg_pos', 'real_silh_sd_pos',
                                        'real_silh_avg_mix', 'real_silh_sd_mix',
                                        'real_silh_avg_neg', 'real_silh_sd_neg',
                                        'p_perm_total', 'p_perm_neu', 'p_perm_pos',
                                        'p_perm_mix', 'p_perm_neg',
                                        'avg_d_neu', 'avg_d_pos', 'avg_d_mix', 'avg_d_neg',
                                        'med_d_neu', 'med_d_pos', 'med_d_mix', 'med_d_neg'])

### ANALYSIS LOOP
for clip in clip_list:
    
    ### STEP 1. LOAD DATASET
    tmp_csv_name = target_sub + "_" + clip + "-STFT-CEBRA.csv"
    tmp_csv = pd.read_csv(os.path.join(data_dir, target_sub, 'CEBRA_input', tmp_csv_name))
    
    tmp_eeg = tmp_csv.iloc[:, :-1].to_numpy()
    tmp_val = tmp_csv.iloc[:, -1].to_numpy()
    
    print(f"STEP 1: {target_sub}'s {clip} dataset was completely loaded!")
    
    
    ### STEP 2. FIT CEBRA MODEL AND SAVE IT
    ## 2-A. fit the model
    tmp_model_real = cebra.CEBRA(batch_size = len(tmp_eeg),               # using the single batch
                                 model_architecture = 'offset10-model',   
                                 num_hidden_units = 38,                   
                                 learning_rate = 1e-3,                    # default value
                                 output_dimension = dim,                  # optimized dimensions
                                 max_iterations = 500,
                                 temperature_mode = 'auto',
                                 device = 'cuda',
                                 hybrid = False,
                                 verbose = False)
    
    tmp_model_real.fit(tmp_eeg, tmp_val) 

    ## 2-B. save the model
    tmp_save_dir = f"../results/cebra_models/canonical_models/{target_sub}"
    os.makedirs(tmp_save_dir, exist_ok = True)
    tmp_model_name = f"{tmp_save_dir}/{target_sub}_{clip}_model_real.pt"
    tmp_model_real.save(tmp_model_name) 

    print(f"STEP 2: The real CEBRA model for {target_sub}'s {clip} dataset were completely trained and saved!")
    
    
    ### STEP 3. CALCULATE SILHOUETTE SCORE
    ## 3-A. load the model
    tmp_model_dir = f"../results/cebra_models/canonical_models/{target_sub}"
    tmp_model_name = f"{tmp_model_dir}/{target_sub}_{clip}_model_real.pt"
    tmp_model_real = cebra.CEBRA.load(tmp_model_name)

    ## 3-B. extract the embeddings
    real_embedding = tmp_model_real.transform(tmp_eeg)
    real_cosine_mtx = cosine_distances(real_embedding)
        
    ## 3-C. compute silhouette scores for each valence label
    real_silh_samples = silhouette_samples(real_cosine_mtx, tmp_val, metric = 'precomputed')
    real_avg_silh_total = silhouette_score(real_cosine_mtx, tmp_val, metric = 'precomputed')
    real_sd_silh_total = np.std(real_silh_samples)
        
    real_silh_result_df = pd.DataFrame({'valence': tmp_val.flatten(), 'silhouettes': real_silh_samples.flatten()})
    real_silh_result_df['valence'] = real_silh_result_df['valence'].astype('category')
    real_silh_result_df['valence'] = real_silh_result_df['valence'].cat.set_categories([0, 1, 2, 3]) # unify the valence set
        
    real_silh_stats_labels = real_silh_result_df.groupby('valence')['silhouettes'].agg(['mean', 'std'])
        
    # if the dataset doesn't have "A" valence label, its silhouette score will be NaN
    real_avg_silh_neu = real_silh_stats_labels['mean'][0]
    real_sd_silh_neu = real_silh_stats_labels['std'][0]
    real_avg_silh_pos = real_silh_stats_labels['mean'][1]
    real_sd_silh_pos = real_silh_stats_labels['std'][1]
    real_avg_silh_mix = real_silh_stats_labels['mean'][2]
    real_sd_silh_mix = real_silh_stats_labels['std'][2]
    real_avg_silh_neg = real_silh_stats_labels['mean'][3]
    real_sd_silh_neg = real_silh_stats_labels['std'][3]

    print(f"STEP 3: The silhouette scores for {target_sub}'s {clip} data were calculated!")
    
    
    ### STEP 4) PERMUTATION TEST FOR SILHOUETTE SCORES
    perm_silh_total_list = [] 
    perm_silh_neu_list = []       
    perm_silh_pos_list = []       
    perm_silh_mix_list = []       
    perm_silh_neg_list = []   
    
    for idx in range(perm_test_num):
        test_idx = idx + 1
        np.random.seed(test_idx) # fix the seed 
        perm_val = np.random.permutation(tmp_val)
        
        # 4-A. fitting the permuted model
        tmp_model_perm = cebra.CEBRA(batch_size = len(tmp_eeg), 
                                     model_architecture = 'offset10-model',
                                     num_hidden_units = 38,
                                     learning_rate = 1e-3,
                                     output_dimension = dim, 
                                     max_iterations = 500,
                                     temperature_mode = 'auto',
                                     device = 'cuda',
                                     hybrid = False,
                                     verbose = False)
            
        tmp_model_perm.fit(tmp_eeg, perm_val)
    
        # 4-B. save the permuted model
        tmp_model_perm_name = f"{tmp_save_dir}/{target_sub}_{clip}_model_perm_{test_idx:04d}.pt"
        tmp_model_perm.save(tmp_model_perm_name) 
    
        # 4-C. load the permuted model and extract the embedding
        tmp_model_perm_name = f"{tmp_save_dir}/{target_sub}_{clip}_model_perm_{test_idx:04d}.pt"
        tmp_model_perm = cebra.CEBRA.load(tmp_model_perm_name)
        perm_embedding = tmp_model_perm.transform(tmp_eeg)
        perm_cosine_mtx = cosine_distances(perm_embedding)
    
        # 4-D. calculate the permuted silhouette scores 
        perm_silh_samples = silhouette_samples(perm_cosine_mtx, perm_val, metric = 'precomputed')
        perm_silh_result_df = pd.DataFrame({'valence': perm_val.flatten(), 'silhouettes': perm_silh_samples.flatten()})
            
        perm_silh_result_df['valence'] = perm_silh_result_df['valence'].astype('category')
        perm_silh_result_df['valence'] = perm_silh_result_df['valence'].cat.set_categories([0, 1, 2, 3])
            
        perm_silh_stats_labels = perm_silh_result_df.groupby('valence')['silhouettes'].agg(['mean', 'std'])
            
        perm_avg_silh_total = silhouette_score(perm_cosine_mtx, perm_val, metric = 'precomputed')
        perm_silh_total_list.append(perm_avg_silh_total)
            
        perm_avg_silh_neu = perm_silh_stats_labels['mean'][0]
        perm_silh_neu_list.append(perm_avg_silh_neu)
        perm_avg_silh_pos = perm_silh_stats_labels['mean'][1]
        perm_silh_pos_list.append(perm_avg_silh_pos)
        perm_avg_silh_mix = perm_silh_stats_labels['mean'][2]
        perm_silh_mix_list.append(perm_avg_silh_mix)
        perm_avg_silh_neg = perm_silh_stats_labels['mean'][3]
        perm_silh_neg_list.append(perm_avg_silh_neg)
            
        test_iter_idx = '{0:04d}'.format(test_idx)
        print(f'>>> [{test_iter_idx}/{perm_test_num}] Permutation test done.')
    
    # 4-E. summary of permutation test
    perm_silh_summary_df = pd.DataFrame({'perm_total': perm_silh_total_list,
                                         'perm_neutral': perm_silh_neu_list,
                                         'perm_positive': perm_silh_pos_list,
                                         'perm_mixed': perm_silh_mix_list,
                                         'perm_negative': perm_silh_neg_list})

    perm_silh_summary_df.to_csv(f"../results/silhouette_scores/{target_sub}_{clip}_perm_silh_scores.csv")
    
    p_perm_total = sum_with_nan(perm_silh_total_list, real_avg_silh_total) / perm_test_num
    p_perm_neu = sum_with_nan(perm_silh_neu_list, real_avg_silh_neu) / perm_test_num
    p_perm_pos = sum_with_nan(perm_silh_pos_list, real_avg_silh_pos) / perm_test_num
    p_perm_mix = sum_with_nan(perm_silh_mix_list, real_avg_silh_mix) / perm_test_num
    p_perm_neg = sum_with_nan(perm_silh_neg_list, real_avg_silh_neg) / perm_test_num

    print(f"STEP 4: {perm_test_num} times of permutation test for {target_sub}'s {clip} data was completed!")
    
    
    ### STEP 5) CORTICAL DISTINCTIVENESS (AVERAGE & MEDOID)
    neu_embed = real_embedding[tmp_val.flatten() == 0]
    pos_embed = real_embedding[tmp_val.flatten() == 1]
    mix_embed = real_embedding[tmp_val.flatten() == 2]
    neg_embed = real_embedding[tmp_val.flatten() == 3]

    avg_d_neu = avg_d_calculator(neu_embed, pos_embed, mix_embed, neg_embed)
    avg_d_pos = avg_d_calculator(pos_embed, neu_embed, mix_embed, neg_embed)
    avg_d_mix = avg_d_calculator(mix_embed, neu_embed, pos_embed, neg_embed)
    avg_d_neg = avg_d_calculator(neg_embed, neu_embed, pos_embed, mix_embed)

    med_d_neu = med_d_calculator(neu_embed, pos_embed, mix_embed, neg_embed)
    med_d_pos = med_d_calculator(pos_embed, neu_embed, mix_embed, neg_embed)
    med_d_mix = med_d_calculator(mix_embed, neu_embed, pos_embed, neg_embed)
    med_d_neg = med_d_calculator(neg_embed, neu_embed, pos_embed, mix_embed)
        
    print(f"STEP 5: The cortical distinctiveness of all valence lables from {target_sub}'s {clip} data was calculated!")
    
    
    ### STEP 6) AGGREGATE ALL RESULTS
    tmp_appended_row = pd.DataFrame([{'subjectkey': target_sub,
                                      'clip': clip,
                                      'dim': dim,
                                      # silhouette score
                                      'real_silh_avg_total': real_avg_silh_total,
                                      'real_silh_sd_total': real_sd_silh_total,
                                      'real_silh_avg_neu': real_avg_silh_neu,
                                      'real_silh_sd_neu': real_sd_silh_neu, 
                                      'real_silh_avg_pos': real_avg_silh_pos,
                                      'real_silh_sd_pos': real_sd_silh_pos,
                                      'real_silh_avg_mix': real_avg_silh_mix,
                                      'real_silh_sd_mix': real_sd_silh_mix,
                                      'real_silh_avg_neg': real_avg_silh_neg,
                                      'real_silh_sd_neg': real_sd_silh_neg,
                                      # p-perm of silhouette score
                                      'p_perm_total': p_perm_total,
                                      'p_perm_neu': p_perm_neu,
                                      'p_perm_pos': p_perm_pos,
                                      'p_perm_mix': p_perm_mix,
                                      'p_perm_neg': p_perm_neg,
                                      # cortical distinctiveness (average)
                                      'avg_d_neu': avg_d_neu,
                                      'avg_d_pos': avg_d_pos,
                                      'avg_d_mix': avg_d_mix,
                                      'avg_d_neg': avg_d_neg,
                                      # cortical distinctiveness (medoid)
                                      'med_d_neu': med_d_neu,
                                      'med_d_pos': med_d_pos,
                                      'med_d_mix': med_d_mix,
                                      'med_d_neg': med_d_neg}])
        
    target_sub_df = pd.concat([target_sub_df, tmp_appended_row])
    
    print(f"@@@ All ANALYSES FROM {target_sub}'S {clip} DATA WAS COMPLETED! @@@")

target_sub_df.to_csv(f"../results/latent_features/summary_csv/{target_sub}_total-summary.csv")
