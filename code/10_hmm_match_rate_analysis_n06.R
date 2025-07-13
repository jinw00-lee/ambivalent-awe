# ____________________________________________________________________________________________
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# _________________________________
# 10 hmm_match_rate_analysis_n06.R
# _________________________________
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# _________________________________
# July, 2025
# ____________________________________________________________________________________________
# [MATERIALS CREATED BY THIS SCRIPT] FIG 6, STABLE 5

library(tidyr)
library(dplyr)
library(ggridges)
library(ggplot2)

### ________________________________________________________ ####
### @@@ PART I. CHANNEL-LEVEL ANALYSIS @@@ ####
## > I-1. Data Preparation ####
results.total.ch <- read.csv("../results/hmm_match_rate/hmm_match_rate_channel.csv")
results.total.ch$subjectkey <- as.factor(results.total.ch$subjectkey)
results.total.ch$clip <- as.factor(results.total.ch$clip)
results.total.ch$feature <- as.factor(results.total.ch$feature)

feature_list.ch <- as.vector(results.total.ch$feature[c(1:length(levels(results.total.ch$feature)))])
clip_list <- c("SP", "CI", "MO")

## > I-2. Aggregate the null distribution ####
hmm_result_df.ch <- data.frame(
  feature = character(),   
  match_rate = numeric(),
  mean_diff = numeric(),   
  p_value = numeric(),    
  null_value = numeric()  
)

iter_idx = 0

for (feat in feature_list.ch) {
  iter_idx = iter_idx + 1
  
  tmp_df <- results.total.ch[results.total.ch$feature == feat, ]
  
  # real mean difference value
  tmp_df <- tmp_df %>%
    mutate(real_mean_diff = match_rate - rowMeans(select(., starts_with("perm_")), na.rm = TRUE))
  
  avg_match_rate <- mean(tmp_df$match_rate)
  avg_real_mean_diff <- mean(tmp_df$real_mean_diff)
  
  # permutation test
  avg_perm_mean_diff <- c()
  
  for (perm_idx in c(1:1000)) {
    tmp_perm <- paste0("perm_", sprintf("%04d", perm_idx))
    tmp_perm_mean_diff <- tmp_df[, tmp_perm] - rowMeans(tmp_df %>% select(starts_with("perm_")), na.rm = TRUE)
    tmp_avg_mean_diff <- mean(as.vector(tmp_perm_mean_diff))
    
    avg_perm_mean_diff <- c(avg_perm_mean_diff, tmp_avg_mean_diff)
  }
  
  feat_p_value = sum(avg_perm_mean_diff > avg_real_mean_diff) / length(avg_perm_mean_diff)
  
  appended_rows <- data.frame(
    feature = rep(feat, 1000),
    match_rate = rep(avg_match_rate, 1000),
    mean_diff = rep(avg_real_mean_diff, 1000),
    p_value = rep(feat_p_value, 1000),
    null_value = avg_perm_mean_diff
  )
  
  hmm_result_df.ch <- rbind(hmm_result_df.ch, appended_rows)
  print(paste0("[", iter_idx, "/", length(feature_list.ch), "] analysis for ", feat, " was done: match rate = ", 
               round(avg_match_rate, 3), " (p = ", round(feat_p_value, 3), ")"))
}

rm(feat, iter_idx, tmp_df, avg_real_mean_diff, avg_match_rate, avg_perm_mean_diff, perm_idx, tmp_perm, 
   tmp_perm_mean_diff, tmp_avg_mean_diff, feat_p_value, appended_rows)

hmm_result_df.ch$feature <- as.factor(hmm_result_df.ch$feature)


## > I-3. Visualization (FIG 6A & STABLE 5) ####
unique_meandiff.ch <- hmm_result_df.ch %>% distinct(feature, match_rate, mean_diff, p_value) #STable 5

fig6a <- ggplot(hmm_result_df.ch, aes(x = null_value, y = feature)) +
  geom_density_ridges(
    rel_min_height = 0.01, scale = 1.2, 
    color = "white"
  ) +
  geom_segment(data = unique_meandiff.ch, 
               aes(x = mean_diff, xend = mean_diff,
                   y = as.numeric(feature), yend = as.numeric(feature) + 0.5),
               color = "black", size = 0.5) +
  theme(
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    axis.text.x = element_text(angle = 45, hjust = 1) 
  ) +
  coord_flip()


### ________________________________________________________ ####
### @@@ PART II. BAND-LEVEL ANALYSIS @@@ ####
## > II-1. Data Preparation ####
results.total.fq <- read.csv("../results/hmm_match_rate/hmm_match_rate_freqband.csv")
results.total.fq$subjectkey <- as.factor(results.total.fq$subjectkey)
results.total.fq$clip <- as.factor(results.total.fq$clip)
results.total.fq$feature <- as.factor(results.total.fq$feature)

feature_list.fq <- as.vector(results.total.fq$feature[c(1:length(levels(results.total.fq$feature)))])
clip_list <- c("SP", "CI", "MO")

## > II-2. Aggregate the Null Distribution ####
hmm_result_df.fq <- data.frame(
  feature = character(),   
  match_rate = numeric(),
  mean_diff = numeric(),   
  p_value = numeric(),    
  null_value = numeric()  
)

iter_idx = 0

for (feat in feature_list.fq) {
  iter_idx = iter_idx + 1
  
  tmp_df <- results.total.fq[results.total.fq$feature == feat, ]
  
  # real mean difference value
  tmp_df <- tmp_df %>%
    mutate(real_mean_diff = match_rate - rowMeans(select(., starts_with("perm_")), na.rm = TRUE))
  
  avg_match_rate <- mean(tmp_df$match_rate)
  avg_real_mean_diff <- mean(tmp_df$real_mean_diff)
  
  # permutation test
  avg_perm_mean_diff <- c()
  
  for (perm_idx in c(1:1000)) {
    tmp_perm <- paste0("perm_", sprintf("%04d", perm_idx))
    tmp_perm_mean_diff <- tmp_df[, tmp_perm] - rowMeans(tmp_df %>% select(starts_with("perm_")), na.rm = TRUE)
    tmp_avg_mean_diff <- mean(as.vector(tmp_perm_mean_diff))
    
    avg_perm_mean_diff <- c(avg_perm_mean_diff, tmp_avg_mean_diff)
  }
  
  feat_p_value = sum(avg_perm_mean_diff > avg_real_mean_diff) / length(avg_perm_mean_diff)
  
  appended_rows <- data.frame(
    feature = rep(feat, 1000),
    match_rate = rep(avg_match_rate, 1000),
    mean_diff = rep(avg_real_mean_diff, 1000),
    p_value = rep(feat_p_value, 1000),
    null_value = avg_perm_mean_diff
  )
  
  hmm_result_df.fq <- rbind(hmm_result_df.fq, appended_rows)
  print(paste0("[", iter_idx, "/", length(feature_list.fq), "] analysis for ", feat, " was done: match rate = ", 
               round(avg_match_rate, 3), " (p = ", round(feat_p_value, 3), ")"))
}

rm(feat, iter_idx, tmp_df, avg_real_mean_diff, avg_match_rate, avg_perm_mean_diff, perm_idx, tmp_perm, 
   tmp_perm_mean_diff, tmp_avg_mean_diff, feat_p_value, appended_rows)

hmm_result_df.fq$feature <- as.factor(hmm_result_df.fq$feature)


## > II-3. Visualization (FIG 6B) ####
unique_meandiff.fq <- hmm_result_df.fq %>% distinct(feature, match_rate, mean_diff, p_value)

fig6b <- ggplot(hmm_result_df.fq, aes(x = null_value, y = factor(feature, levels = c("delta", "theta", "alpha", "beta", "gamma")))) +   
  geom_density_ridges(rel_min_height = 0.01, scale = 1.20, color = "white") +   
  geom_segment(data = unique_meandiff.fq,
               aes(x = mean_diff, xend = mean_diff, 
                   y = as.numeric(factor(feature, levels = c("delta", "theta", "alpha", "beta", "gamma"))), 
                   yend = as.numeric(factor(feature, levels = c("delta", "theta", "alpha", "beta", "gamma"))) + 0.5),
               color = "black", size = 0.5) +   
  theme(
    panel.border = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    axis.line = element_line(colour = "black"),
    axis.text.x = element_text(angle = 45, hjust = 1)
  ) +   
  coord_flip()

fig6b


### ________________________________________________________ ####
### RESULTS ####
fig6a

fig6b
