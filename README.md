# Awe is characterized as an ambivalent emotion in the human behavior and cortex  
**Jinwoo Lee**, Danny Dongyeop Han, Seung-Yeop Oh, & **Jiook Cha**.   
Seoul National University   
- First Author: Jinwoo Lee (adem1997@snu.ac.kr | [jinwoo-lee.com](https://jinwoo-lee.com))   
- Corresponding Author: Jiook Cha (connectome@snu.ac.kr | [www.connectomelab.com](https://www.connectomelab.com))
- Preprint: [https://doi.org/10.1101/2024.08.18.608520](https://doi.org/10.1101/2024.08.18.608520)

## Abstract
Awe is a complex emotion encompassing both positive and negative feelings, but its ambivalent affect remains underexplored. To address whether and how awe's ambivalence is represented both behaviorally and neurologically, we conducted a study using virtual reality and electroencephalography (*N* = 43). Behaviorally, the subjective intensity of awe was best predicted by the duration and intensity of ambivalent feelings, not by single valence-related metrics. In the electrophysiological analysis, we identified latent neural spaces for each participant that shared valence representations across individuals and stimuli, using a deep representational learning. Within these spaces, ambivalent feelings were represented as spatially distinct from positive and negative ones. Notably, their degree of separation specifically predicted awe ratings. Lastly, frontal delta band power mainly differentiated valence types. Our findings underline ambivalence of awe, which integrates competing affects. This work provides a nuanced framework for understanding human emotions with implications for affective neuroscience and relevant fields such as mental wellness.

![AweVR_video_abstract](https://github.com/user-attachments/assets/c122bc4e-7af7-497e-ab87-9682a529ebab)


## Guided Tour for Scripts   
In general, we utilized three programming languages in this study for distinct purposes:
- **Matlab**: Preprocessing of behavioral and EEG data (except for STFT). 
- **Python**: CEBRA-based decoding analysis, XAI analysis (i.e., Dynamask and HMM).
- **R**: Predictive modeling, statistical analysis, and visualization.   
   
In the following section, we will explain how each code was utilized in different stages of analysis.   
### Part I. Data Preprocessing ###
- **(01) `eeg_preprocessing_1.m`**: This code performs EEG signal epoching and artifact removal using EEGLAB, following [Delorme (2024)](https://www.nature.com/articles/s41598-023-27528-0). It applies ASR cleaning and ICA-based component rejection for this process.

   ---
   > **NOTE** You should install EEGLAB Plugin first to handle Neuroelectrics' `.easy` data format in this script. Please check [here](https://www.neuroelectrics.com/eeglab-plugin).
   ---
   
- **(02) `eeg_preprocessing_2.m`**: This code normalizes preprocessed EEG signals during VR watching. It then performs interpolation to temporally align valence keypress event markers with EEG samples. Finally, it generates a CSV file containing normalized values for 19 channels, latency, and valence labels for each timepoint.
- **(03) `eeg_valence_STFT.py`**: This code applies a STFT to the EEG signals in the CSV dataframe, extracting five frequency band power features for each channel within a 1-second time window. The valence label for each time window is defined as the mode of the valence labels within that window.

### Part II. CEBRA-based Pairwise Decoding and Attribution ###
- **(04) `cebra_embedding_learning.py`**: This code trains CEBRA on the STFT-transformed EEG and valence sequences for each participant and video. Latent embeddings are sequentially learned from 1 to 9 dimensions. Additionally, for each embedding, a permutation test is conducted 1,000 times by randomly shuffling the valence sequence and learning embeddings using CEBRA. The following metrics are calculated for each valence type:   

  - a) Silhouette score         
  - b) Permutation-based p-value of the silhouette score         
  - c) Cortical distinctiveness based on average and medoid distances.         
  ---
  > **NOTE** *Running this code takes a significant amount of time. Therefore, it is recommended to use job schedulers like SLURM to run the code in parallel, either by video or by participant.*
  ---   

- **(05) `cebra_decoding.py`**: This code performs pairwise decoding analysis using data from six participants who reported all four valence labels across three awe conditions. Based on the previously learned 1-9 dimensional CEBRA embeddings, decoding analyses are conducted under the 'across participants' and 'across clips' tasks. For each task, decoding is performed under three conditions: 'random', 'not aligned', and 'aligned'. CCA is used for the aligned condition analysis. For each decoding task, a `.csv` dataframe is saved, containing the train set, test set, dimension, and test performance for each condition.
- **(06) `pca_decoding.py`**: This code performs a baseline analysis for the CEBRA decoding analysis by conducting pairwise decoding analysis using PCA embeddings in the same manner. The results are saved in a `.csv` dataframe.   
- **(07) `faa_decoding.py`**: This code performs a baseline analysis for the CEBRA decoding analysis by conducting pairwise decoding analysis using FAA embeddings (i.e., the alpha power difference between F4 and F3 at each timepoint). The results are saved in a `.csv` dataframe.
- **(08) `dynamask_implement.py`**: This code calls the saved CEBRA models back into Dynamask and trains attribution maps to determine which features and timepoints were most important for the embedding learning process in each participant-video dataset.
- **(09) `hmm_implement.py`**: This code uses HMM to evaluate how well the latent state transitions of each frequency band feature align with the time-based transitions of the actual keypress valence sequence. A transition is considered a "match" if it occurs within 3 seconds of the actual keypress transition. A permutation test is conducted 1,000 times to calculate the actual match rate and the match rates for each permutation test.

### Part III. Predictive Modeling, Statistical Analysis, and Visualization ###
- **(10) `behavior_analysis.R`**: This code performs the following tasks:
       
   - a) investigates the psychometric validity of self-report data.     
   - b) tests statistical differences in affective features across clip conditions. &rarr; `Fig 2`
   - c) conducts univariate and multivariate predictive modeling to explain awe ratings based on behavioral variables. &rarr; `Fig 3`     
      - cf.) `Fig 1` *was created manually.*     

- **(11) `latent_space_analysis.R`**: This code performs the following tasks:

   - a) conducts dimensionality selection for CEBRA- and PCA-driven spaces with decoding performances. &rarr; `Fig 4a` & `SFig 3`
   - b) examines statistical differences in decoding performances across tasks and analytic tools. &rarr; `Fig 4b`  & `SFig 4`
   - c) displays silhoette score and its statistical significance for each valence category. &rarr; `Fig 5a`
        - cf.) `Fig 5b` *was created manually.*    
   - d) predicts self-reported awe scores with cortical distinctiveness value. &rarr; `Fig 5c-e`
 
- **(12) `dynamask_weight_analysis.R`**: This code performs the following tasks:

   - a) calculates average perturbation weights of each STFT feature per valence category. &rarr; `Fig 6a-b` & `SFig 6`
   - b) conducts hierarchical clustering analysis for average perturbation weights of ambivalent states. &rarr; `SFig 5`
 
- **(13) `hmm_match_rate_analysis.R`**: This code performs the following tasks:

   - a) evaluates statistical significance of HMM-based match rate through permutation tests. &rarr; `Fig 6c`
 
### Supplementary: Video-Sensory Features Analysis ###
- **(14) `sensory_feature_extraction.py`**: This code extracts and calculates the brightness, color hue, contrast of the video, and the loudness of the audio every 0.5 seconds from a 2D video reconstructed from the frontal perspective of a 3D VR video.
- **(15) `sensory_feature_analysis.R`**: This code visualizes the temporal patterns of four sensory features for each clip and tests whether each feature is significantly correlated with an individual's valence sequence at each time point. &rarr; `SFig 1`

## Citation
```
@article{yi2024awe,
  title={Awe is characterized as an ambivalent experience in the human behavior and cortex: integrated virtual reality-electroencephalogram study},
  author={Yi, Jinwoo and Han, Danny Dongyeop and Oh, Seung-Yeop and Cha, Jiook},
  journal={bioRxiv},
  pages={2024--08},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```
