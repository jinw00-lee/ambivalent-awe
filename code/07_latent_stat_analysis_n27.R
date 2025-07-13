# ____________________________________________________________________________________________
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# _________________________________
# 07 latent_stat_analysis_n27.R
# _________________________________
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# _________________________________
# July, 2025
# ____________________________________________________________________________________________
# [MATERIALS CREATED BY THIS SCRIPT] FIG 5, SFIG 3

library(readxl)
library(dplyr)
library(purrr)
library(readr)
library(reshape2)
library(lmerTest)
library(ggplot2)
library(h2o)
library(bayestestR)

### ________________________________________________________ ####
### @@@ I. DATA LOADING AND FORMATTING @@@ ####
sub.list <- read.csv("../dataset/sub-list-latent_n27.csv", header = FALSE)
sub.list <- unlist(as.vector(sub.list[1]), use.names = FALSE)
clip.list <- c("SP", "CI", "MO")

## > I-1. Pre-experiment Survey ####
pre.total.df <- read_excel("../dataset/behavior/pre-experiment-survey_n43.xlsx", "pre-experiment-survey_n43")
pre.total.df <- pre.total.df[pre.total.df$subjectkey %in% sub.list, 
                             c("subjectkey", "sex_M", "age", "panas_p_total", "panas_n_total", "dpes_awe_total")]

## > I-2. Post-experiment Ratings ####
self.SP <- read_excel("../dataset/behavior/report_SP_n43.xlsx", "report_SP_n43")
self.CI <- read_excel("../dataset/behavior/report_CI_n43.xlsx", "report_CI_n43")
self.MO <- read_excel("../dataset/behavior/report_MO_n43.xlsx", "report_MO_n43")

beh.total.df <- rbind(self.SP, self.CI, self.MO)
beh.total.df <- beh.total.df[beh.total.df$subjectkey %in% sub.list, ]

beh.total.df$subjectkey <- as.factor(beh.total.df$subjectkey)
beh.total.df$clip <- as.factor(beh.total.df$clip)
beh.total.df$order <- as.factor(beh.total.df$order)
rm(self.SP, self.CI, self.MO) # clean the environment

## > I-3. Keypress Dataset ####
len_faa.total.df <- data.frame(matrix(nrow = 0 , ncol = 6))

for (sub_idx in c(1:length(sub.list))) {
  for (clip_idx in c(1:length(clip.list))) {
    
    # search preprocessed EEG-Valence paired dataset
    tmp_sub <- sub.list[sub_idx]
    tmp_clip <- clip.list[clip_idx]
    tmp_filename <- paste0(tmp_sub, "_", tmp_clip, "-STFT-CEBRA.csv")
    tmp_dir <- paste0("../dataset/eeg/pped/", tmp_sub, "/CEBRA_input/", tmp_filename)
    tmp_file <- read.csv(tmp_dir, header = TRUE)
    
    # get the subject's keypress vector
    tmp_file$mode <- as.factor(tmp_file$mode)
    mode_table <- as.data.frame(table(tmp_file$mode))
    len_neu <- mode_table[mode_table$Var1 == '0', 'Freq'] / length(tmp_file$mode)
    if (length(len_neu) == 0) {len_neu <- 0}
    len_pos <- mode_table[mode_table$Var1 == '1', 'Freq'] / length(tmp_file$mode)
    if (length(len_pos) == 0) {len_pos <- 0}
    len_mix <- mode_table[mode_table$Var1 == '2', 'Freq'] / length(tmp_file$mode)
    if (length(len_mix) == 0) {len_mix <- 0}
    len_neg <- mode_table[mode_table$Var1 == '3', 'Freq'] / length(tmp_file$mode)
    if (length(len_neg) == 0) {len_neg <- 0}
    
    # get the time-averaged FAA value
    avg_faa <- mean(tmp_file[, "alpha_F4"] - tmp_file[, "alpha_F3"])
    
    len_faa.total.df <- rbind(len_faa.total.df, c(tmp_sub, tmp_clip, len_neu, len_pos, len_mix, len_neg, avg_faa))
  }
  
  # verbose
  print(paste0("[", sub_idx, "/", length(sub.list), "] The keypress features were copied successfully to all clip datasets!"))
}

# clean the environment
rm(sub_idx, clip_idx, tmp_sub, tmp_clip, tmp_filename, tmp_dir, tmp_file, mode_table, len_neu, len_pos, len_mix, len_neg, avg_faa)
colnames(len_faa.total.df) <- c("subjectkey", "clip", "len_neu", "len_pos", "len_mix", "len_neg", "avg_faa")

## > I-4. Latent Feature Dataset ####
csv.paths <- file.path("../results/latent_features/summary_csv", paste0(sub.list, "_total-summary.csv"))

lat.total.df <- map_dfr(csv.paths, read_csv, .id = "source")
lat.total.df <- lat.total.df[, c(3:ncol(lat.total.df))] # exclude the first two non-informative columns 
lat.total.df$subjectkey <- as.factor(lat.total.df$subjectkey)
lat.total.df$clip <- as.factor(lat.total.df$clip)

rm(csv.paths) # clean the environment

## > I-5. Merge the Datasets ####
beh.total.df <- beh.total.df %>% arrange(subjectkey)
lat.total.df <- lat.total.df %>% arrange(subjectkey)
len_faa.total.df <- len_faa.total.df %>% arrange(subjectkey)

total.df <- merge(beh.total.df, lat.total.df, by = c('subjectkey', 'clip'))
total.df <- merge(total.df, len_faa.total.df, by = c('subjectkey', 'clip'))
total.df <- pre.total.df %>% left_join(total.df, by = 'subjectkey')

total.df$subjectkey <- as.factor(total.df$subjectkey)
total.df$len_neu <- as.numeric(total.df$len_neu)
total.df$len_pos <- as.numeric(total.df$len_pos)
total.df$len_mix <- as.numeric(total.df$len_mix)
total.df$len_neg <- as.numeric(total.df$len_neg)
total.df$avg_faa <- as.numeric(total.df$avg_faa)

rm(beh.total.df, lat.total.df, len_faa.total.df, pre.total.df) # clean the environment

### ________________________________________________________ ####
### @@@ II. VARIABILITY IN SILHOUETTE SCORES S [FIG 5A & SFIG 3] @@@ ####
## > II-1. Ambivalent States [FIG 5A] ####
# >> II-1-a. data preparation ####
silh.mix.df <- data.frame(subjectkey = factor(), SP = numeric(), CI = numeric(), MO = numeric()) 
pval.mix.df <- data.frame(subjectkey = factor(), SP = numeric(), CI = numeric(), MO = numeric())

for (sub in sub.list) {
  add.silh.row <- c()
  add.pval.row <- c()
  
  for (clip in clip.list) {
    tmp.df <- total.df[total.df$subjectkey == sub & total.df$clip == clip, ]
    
    tmp.silh <- tmp.df$real_silh_avg_mix
    tmp.p <- tmp.df$p_perm_mix
    
    add.silh.row <- append(add.silh.row, tmp.silh)
    add.pval.row <- append(add.pval.row, tmp.p)
  }
  
  silh.mix.df <- rbind(silh.mix.df, c(sub, add.silh.row))
  pval.mix.df <- rbind(pval.mix.df, c(sub, add.pval.row))
}

rm(sub, add.silh.row, add.pval.row, clip, tmp.df, tmp.silh, tmp.p) # clean environment

colnames(silh.mix.df) <- c("subjectkey", "SP", "CI", "MO")
silh.mix.df$subjectkey <- as.factor(silh.mix.df$subjectkey)
silh.mix.df$SP <- as.numeric(silh.mix.df$SP)
silh.mix.df$CI <- as.numeric(silh.mix.df$CI)
silh.mix.df$MO <- as.numeric(silh.mix.df$MO)

colnames(pval.mix.df) <- c("subjectkey", "SP", "CI", "MO")
pval.mix.df$subjectkey <- as.factor(pval.mix.df$subjectkey)
pval.mix.df$SP <- as.numeric(pval.mix.df$SP)
pval.mix.df$CI <- as.numeric(pval.mix.df$CI)
pval.mix.df$MO <- as.numeric(pval.mix.df$MO)

silh.mix.melted <- melt(silh.mix.df, varnames = c("Row", "Column"))
colnames(silh.mix.melted) <- c("subjectkey", "clip", "silh_avg_mix")
pval.mix.melted <- melt(pval.mix.df, varnames = c("Row", "Column"))
colnames(pval.mix.melted) <- c("subjectkey", "clip", "p_perm_mix")

total.mix.melted <- merge(silh.mix.melted, pval.mix.melted, by = c("subjectkey", "clip"))
rm(silh.mix.df, silh.mix.melted, pval.mix.melted, pval.mix.df) # clean the environment

total.mix.melted$p_perm_mix_fdr <- p.adjust(total.mix.melted$p_perm_mix, method = "fdr")

total.mix.melted$signif <- ifelse(total.mix.melted$p_perm_mix_fdr < 0.001, '***', 
                                  ifelse(total.mix.melted$p_perm_mix_fdr < 0.01, '**', 
                                         ifelse(total.mix.melted$p_perm_mix_fdr < 0.05, '*', 'ns')))

total.mix.melted$signif <- as.factor(total.mix.melted$signif)

# >> II-1-b. visualization ####
fig5a <- ggplot(total.mix.melted, aes(x = subjectkey, y = clip, fill = signif, label = round(silh_avg_mix, 2))) +
  geom_tile() +  
  geom_text(color = "black", size = 3) +  
  scale_fill_manual(values = c('ns' = 'white', 
                               '*' = '#F3E5F5', 
                               '**' = '#E1BEE7', 
                               '***' = '#CE93D8')) +
  labs(x = NULL, y = NULL, fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank()) 

## > II-2. Positive States [SFIG 3] ####
# >> II-2-a. data preparation ####
silh.pos.df <- data.frame(subjectkey = factor(), SP = numeric(), CI = numeric(), MO = numeric()) 
pval.pos.df <- data.frame(subjectkey = factor(), SP = numeric(), CI = numeric(), MO = numeric())

for (sub in sub.list) {
  add.silh.row <- c()
  add.pval.row <- c()
  
  for (clip in clip.list) {
    tmp.df <- total.df[total.df$subjectkey == sub & total.df$clip == clip, ]
    
    tmp.silh <- tmp.df$real_silh_avg_pos
    tmp.p <- tmp.df$p_perm_pos
    
    add.silh.row <- append(add.silh.row, tmp.silh)
    add.pval.row <- append(add.pval.row, tmp.p)
  }
  
  silh.pos.df <- rbind(silh.pos.df, c(sub, add.silh.row))
  pval.pos.df <- rbind(pval.pos.df, c(sub, add.pval.row))
}

rm(sub, add.silh.row, add.pval.row, clip, tmp.df, tmp.silh, tmp.p) # clean environment

colnames(silh.pos.df) <- c("subjectkey", "SP", "CI", "MO")
silh.pos.df$subjectkey <- as.factor(silh.pos.df$subjectkey)
silh.pos.df$SP <- as.numeric(silh.pos.df$SP)
silh.pos.df$CI <- as.numeric(silh.pos.df$CI)
silh.pos.df$MO <- as.numeric(silh.pos.df$MO)

colnames(pval.pos.df) <- c("subjectkey", "SP", "CI", "MO")
pval.pos.df$subjectkey <- as.factor(pval.pos.df$subjectkey)
pval.pos.df$SP <- as.numeric(pval.pos.df$SP)
pval.pos.df$CI <- as.numeric(pval.pos.df$CI)
pval.pos.df$MO <- as.numeric(pval.pos.df$MO)

silh.pos.melted <- melt(silh.pos.df, varnames = c("Row", "Column"))
colnames(silh.pos.melted) <- c("subjectkey", "clip", "silh_avg_pos")
pval.pos.melted <- melt(pval.pos.df, varnames = c("Row", "Column"))
colnames(pval.pos.melted) <- c("subjectkey", "clip", "p_perm_pos")

total.pos.melted <- merge(silh.pos.melted, pval.pos.melted, by = c("subjectkey", "clip"))
rm(silh.pos.df, silh.pos.melted, pval.pos.melted, pval.pos.df) # clean the environment

total.pos.melted$p_perm_pos_fdr <- p.adjust(total.pos.melted$p_perm_pos, method = "fdr")

total.pos.melted$signif <- ifelse(total.pos.melted$p_perm_pos_fdr < 0.001, '***', 
                                  ifelse(total.pos.melted$p_perm_pos_fdr < 0.01, '**', 
                                         ifelse(total.pos.melted$p_perm_pos_fdr < 0.05, '*', 'ns')))

total.pos.melted$signif <- as.factor(total.pos.melted$signif)

# >> II-2-b. visualization ####
sfig3.pos <- ggplot(total.pos.melted, aes(x = subjectkey, y = clip, fill = signif, label = round(silh_avg_pos, 2))) +
  geom_tile() +  
  geom_text(color = "black", size = 3) +  
  scale_fill_manual(values = c('ns' = 'white', 
                               '*' = '#FFEBEE', 
                               '**' = '#FFCDD2', 
                               '***' = '#EF9A9A')) +
  labs(x = NULL, y = NULL, fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank()) 

## > II-3. Negative States [SFIG 3] ####
# >> II-3-a. data preparation ####
silh.neg.df <- data.frame(subjectkey = factor(), SP = numeric(), CI = numeric(), MO = numeric()) 
pval.neg.df <- data.frame(subjectkey = factor(), SP = numeric(), CI = numeric(), MO = numeric())

for (sub in sub.list) {
  add.silh.row <- c()
  add.pval.row <- c()
  
  for (clip in clip.list) {
    tmp.df <- total.df[total.df$subjectkey == sub & total.df$clip == clip, ]
    
    tmp.silh <- tmp.df$real_silh_avg_neg
    tmp.p <- tmp.df$p_perm_neg
    
    add.silh.row <- append(add.silh.row, tmp.silh)
    add.pval.row <- append(add.pval.row, tmp.p)
  }
  
  silh.neg.df <- rbind(silh.neg.df, c(sub, add.silh.row))
  pval.neg.df <- rbind(pval.neg.df, c(sub, add.pval.row))
}

rm(sub, add.silh.row, add.pval.row, clip, tmp.df, tmp.silh, tmp.p) # clean environment

colnames(silh.neg.df) <- c("subjectkey", "SP", "CI", "MO")
silh.neg.df$subjectkey <- as.factor(silh.neg.df$subjectkey)
silh.neg.df$SP <- as.numeric(silh.neg.df$SP)
silh.neg.df$CI <- as.numeric(silh.neg.df$CI)
silh.neg.df$MO <- as.numeric(silh.neg.df$MO)

colnames(pval.neg.df) <- c("subjectkey", "SP", "CI", "MO")
pval.neg.df$subjectkey <- as.factor(pval.neg.df$subjectkey)
pval.neg.df$SP <- as.numeric(pval.neg.df$SP)
pval.neg.df$CI <- as.numeric(pval.neg.df$CI)
pval.neg.df$MO <- as.numeric(pval.neg.df$MO)

silh.neg.melted <- melt(silh.neg.df, varnames = c("Row", "Column"))
colnames(silh.neg.melted) <- c("subjectkey", "clip", "silh_avg_neg")
pval.neg.melted <- melt(pval.neg.df, varnames = c("Row", "Column"))
colnames(pval.neg.melted) <- c("subjectkey", "clip", "p_perm_neg")

total.neg.melted <- merge(silh.neg.melted, pval.neg.melted, by = c("subjectkey", "clip"))
rm(silh.neg.df, silh.neg.melted, pval.neg.melted, pval.neg.df) # clean the environment

total.neg.melted$p_perm_neg_fdr <- p.adjust(total.neg.melted$p_perm_neg, method = "fdr")

total.neg.melted$signif <- ifelse(total.neg.melted$p_perm_neg_fdr < 0.001, '***', 
                                  ifelse(total.neg.melted$p_perm_neg_fdr < 0.01, '**', 
                                         ifelse(total.neg.melted$p_perm_neg_fdr < 0.05, '*', 'ns')))

total.neg.melted$signif <- as.factor(total.neg.melted$signif)

# >> II-3-b. visualization ####
sfig3.neg <- ggplot(total.neg.melted, aes(x = subjectkey, y = clip, fill = signif, label = round(silh_avg_neg, 2))) +
  geom_tile() +  
  geom_text(color = "black", size = 3) +  
  scale_fill_manual(values = c('ns' = 'white', 
                               '*' = '#E8EAF6', 
                               '**' = '#C5CAE9', 
                               '***' = '#9FA8DA')) +
  labs(x = NULL, y = NULL, fill = "") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title = element_blank()) 


## > II-4. Testing Confounding Effects of Cluster Size ####
size.p.lm_null <- lm(p_perm_mix ~ 1, data = total.df)
size.p.lm_full <- lm(p_perm_mix ~ len_mix, data = total.df)
summary(size.p.lm_full)
bayesfactor_models(size.p.lm_full, size.p.lm_null)

### ________________________________________________________ ####
### @@@ III. CORTICAL DISTINCTIVENESS ANALYSES [FIG 5C-E] @@@ ####
## > III-1. Univariate Analysis ####
predictor.list <- c("avg_d_neu", "avg_d_pos", "avg_d_mix", "avg_d_neg", 
                    "med_d_neu", "med_d_pos", "med_d_mix", "med_d_neg")

res.uni.lmm <- data.frame(matrix(nrow = 0 , ncol = 8))

# >> III-1-a. statistical testing ####
for (idx in c(1:length(predictor.list))) {
  tmp_predictor <- predictor.list[idx]
  tmp_lmm <- lmer(awes_avg ~ total.df[, tmp_predictor][[1]] + (1|subjectkey) + (1|clip), data = total.df)
  tmp_lmm_info <- summary(tmp_lmm)
  
  coef <- tmp_lmm_info$coefficients[[2]]
  se <- tmp_lmm_info$coefficients[[4]]
  ci_lower <- coef - 1.96*se
  ci_upper <- coef + 1.96*se
  df <- tmp_lmm_info$coefficients[[6]]
  t <- tmp_lmm_info$coefficients[[8]]
  p <- tmp_lmm_info$coefficients[[10]]
  
  res.uni.lmm <- rbind(res.uni.lmm, c(tmp_predictor, coef, se, ci_lower, ci_upper, df, t, p))
}

rm(idx, tmp_predictor, tmp_lmm, tmp_lmm_info, coef, se, ci_lower, ci_upper, df, t, p)
colnames(res.uni.lmm) <- c("predictor", "coefficient", "se", "ci_lower", "ci_upper", "df", "t_value", "p_value")

write.csv(res.uni.lmm, "../results/latent_features/univariate_modeling_stats.csv")

for (i in c(2:ncol(res.uni.lmm))) {res.uni.lmm[[i]] <- as.numeric(res.uni.lmm[[i]])}
rm(i)

# >> III-1-b. visualization ####
fig5c <- ggplot(res.uni.lmm, aes(x = coefficient, y = predictor)) +
  geom_point() +
  scale_y_discrete(limits = c("med_d_neg", "med_d_pos", "med_d_neu", "med_d_mix", 
                              "avg_d_neg", "avg_d_pos", "avg_d_neu", "avg_d_mix")) +
  geom_errorbar(aes(xmin = coefficient - 1.96*se, xmax = coefficient + 1.96*se), width = .5) +
  geom_vline(xintercept = 0, linetype = "dotted") +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "white"))

## > III-2. Control Analysis: Time-averaged FAA ####
faa_lmm_null <- lmer(awes_avg ~ 1 + (1|subjectkey) + (1|clip), data = total.df)
faa_lmm_full <- lmer(awes_avg ~ avg_faa + (1|subjectkey) + (1|clip), data = total.df)
summary(faa_lmm_full)
bayesfactor_models(faa_lmm_full, faa_lmm_null)

## > III-3. Multivariate ML Analysis ####
# >> III-3-a. initial setting ####
data.h2o <- total.df[, c("sex_M", "age", "panas_p_total", "panas_n_total", "dpes_awe_total",
                         "awes_avg", "esg_pos", "esg_neg", "esg_eci", "arousal", "motion_sickness",
                         "len_neu", "len_pos", "len_mix", "len_neg", "avg_d_mix")]

h2o.init()
data.h2o <- as.h2o(data.h2o)
h2o.describe(data.h2o)
colnames(data.h2o)

data.h2o.train <- h2o.splitFrame(data.h2o, ratios = 0.8, seed = 1)[[1]]
data.h2o.test <- h2o.splitFrame(data.h2o, ratios = 0.8, seed = 1)[[2]]

h2o.y <- "awes_avg" 
h2o.with_cd.x <- c("sex_M", "age", "panas_p_total", "panas_n_total", "dpes_awe_total", 
                   "esg_pos", "esg_neg", "esg_eci", "arousal", "motion_sickness", 
                   "len_neu", "len_pos", "len_mix", "len_neg", "avg_d_mix")
h2o.no_cd.x <- h2o.with_cd.x[-1]


# >> III-3-b. model fitting (1) - with CD ####
# ______________________________________________________________________________________________________________________
# [NOTE!] To reproduce the results, please load the pre-trained model before proceeding with the analysis. (line 440 ~ )
# ______________________________________________________________________________________________________________________
# automl.with_cd.5cv <- h2o.automl(x = h2o.with_cd.x, 
#                                  y = h2o.y, 
#                                  training_frame = data.h2o.train, 
#                                  max_models = 20, 
#                                  seed = 1,
#                                  nfolds = 5,
#                                  keep_cross_validation_fold_assignment = TRUE,
#                                  keep_cross_validation_predictions = TRUE,
#                                  keep_cross_validation_models = TRUE)

# automl.with_cd.5cv.board <- h2o.get_leaderboard(automl.with_cd.5cv, extra_columns = "ALL")
# automl.with_cd.board.df <- as.data.frame(automl.with_cd.5cv.board)

# >> III-3-c. model fitting (2) - without CD ####
# automl.no_cd.5cv <- h2o.automl(x = h2o.no_cd.x, 
#                                y = h2o.y, 
#                                training_frame = data.h2o.train, 
#                                max_models = 20, 
#                                seed = 1,
#                                nfolds = 5,
#                                keep_cross_validation_fold_assignment = TRUE,
#                                keep_cross_validation_predictions = TRUE,
#                                keep_cross_validation_models = TRUE)

# automl.no_cd.5cv.board <- h2o.get_leaderboard(automl.no_cd.5cv, extra_columns = "ALL")
# automl.no_cd.board.df <- as.data.frame(automl.no_cd.5cv.board)

# >> III-3-d. model selection ####
# automl.with_cd.best.model <- automl.with_cd.board.df[1, "model_id"]  # best model with CD feature
# automl.best.with_cd <- h2o.getModel(automl.with_cd.best.model)
                                                            
# h2o.saveModel(automl.best.with_cd,                               
#               path = "../results/latent_features", force = TRUE)
 
# automl.no_cd.best.model <- automl.no_cd.board.df[1, "model_id"]      # best model without CD feature
# automl.best.no_cd <- h2o.getModel(automl.no_cd.best.model)

# h2o.saveModel(automl.no_cd.best.DL,                               
#               path = "../results/latent_features", force = TRUE)
# ______________________________________________________________________________________________________________________

# >> III-3-e. performance evaluation (Fig 5D) #### 
automl.best.with_cd <- h2o.loadModel("../results/latent_features/autoML_best-model_with-CD")
automl.best.no_cd <- h2o.loadModel("../results/latent_features/autoML_best-model_without-CD")

automl.best.with_cd.perf <- h2o.performance(automl.best.with_cd, xval = T)
automl.best.no_cd.perf <- h2o.performance(automl.best.no_cd, xval = T)

res.h2o.perf.df <- data.frame(matrix(ncol = 3, nrow = 8))

colnames(res.h2o.perf.df) <- c("model", "metric", "value")
res.h2o.perf.df$model <- c(rep("with CD", 4), rep("without CD", 4))
res.h2o.perf.df$metric <- rep(c("MSE", "RMSE", "MAE", "R2"), 2)
res.h2o.perf.df$value <- c(automl.best.with_cd.perf@metrics$MSE, automl.best.with_cd.perf@metrics$RMSE, 
                           automl.best.with_cd.perf@metrics$mae, automl.best.with_cd.perf@metrics$r2,
                           automl.best.no_cd.perf@metrics$MSE, automl.best.no_cd.perf@metrics$RMSE, 
                           automl.best.no_cd.perf@metrics$mae, automl.best.no_cd.perf@metrics$r2)

res.h2o.perf.df$model <- as.factor(res.h2o.perf.df$model)
res.h2o.perf.df$metric <- as.factor(res.h2o.perf.df$metric)
res.h2o.perf.df$value <- as.numeric(res.h2o.perf.df$value)

# visualization
fig5d <- ggplot(res.h2o.perf.df, aes(x = metric, y = value, fill = model)) + 
  geom_col(position = "dodge") + 
  geom_text(aes(label = round(value, 3)), vjust = -0.50, color = "black", size = 3, 
            position = position_dodge(0.9)) +
  scale_fill_manual(values = c("with CD" = "#E1BEE7", "without CD" = "#BDBDBD")) +
  scale_x_discrete(limits = c('R2', 'MAE', 'MSE', 'RMSE')) +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

# >> III-3-f. variable importance (Fig 5E) ####
automl.best.with_cd.varimp <- h2o.varimp(automl.best.with_cd)
automl.best.with_cd.varimp$variable <- reorder(automl.best.with_cd.varimp$variable, 
                                           -automl.best.with_cd.varimp$scaled_importance)

fig5e <- ggplot(automl.best.with_cd.varimp, aes(y = scaled_importance, x = variable)) +
  geom_bar(position = "dodge", stat = "identity", 
           aes(fill = ifelse(variable %in% variable[c(1, 6, 14)], "#E1BEE7", "#BDBDBD")), 
           color = "white") +  
  geom_text(size = 2.50, aes(label = round(scaled_importance, 3)), 
            position = position_dodge(width = 0.90), vjust = 0.3, hjust = -0.2, angle = 90) +  
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"),
        axis.text.x = element_text(angle = 45, vjust = 1, hjust = 1)) +
  scale_fill_identity()

write.csv(res.h2o.perf.df, "../results/latent_features/autoML_performance_table.csv")
write.csv(automl.best.with_cd.varimp, "../results/latent_features/autoML_varimp.csv")


### ________________________________________________________ ####
### RESULTS ####
fig5a

fig5c

fig5d

fig5e

sfig3.pos
sfig3.neg