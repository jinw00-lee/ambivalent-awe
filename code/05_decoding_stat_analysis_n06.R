# ____________________________________________________________________________________________
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# _________________________________
# 05 decoding_stat_analysis_n06.R
# _________________________________
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# _________________________________
# July, 2025
# ____________________________________________________________________________________________
# [MATERIALS CREATED BY THIS SCRIPT] Fig 4, STable 4, SFig 2

# loading required packages
library(tidyr)
library(dplyr)
library(cluster)
library(factoextra)
library(ggplot2)
library(emmeans)
library(lmerTest)

### ________________________________________________________ ####
### @@@ PART I. DIMENSIONALITY SELECTION for CEBRA [FIG 4A] @@@ ####
## > I-1. Data Formatting ####
# >> I-1-a. across-participants ####
acr.sub.cebra.df <- read.csv("../results/decoding_performances/across_sub_decoding_CEBRA.csv")
acr.sub.cebra.df$tool <- "CEBRA"
acr.sub.pca.df <- read.csv("../results/decoding_performances/across_sub_decoding_PCA.csv")
acr.sub.pca.df$tool <- "PCA"
acr.sub.faa.df <- read.csv("../results/decoding_performances/across_sub_decoding_FAA.csv")
acr.sub.faa.df$tool <- "FAA"

acr.sub.total.df <- rbind(acr.sub.cebra.df, acr.sub.pca.df)
acr.sub.total.df <- rbind(acr.sub.total.df, acr.sub.faa.df)

acr.sub.total.df$ref_sub <- as.factor(acr.sub.total.df$ref_sub)
acr.sub.total.df$target_sub <- as.factor(acr.sub.total.df$target_sub)
acr.sub.total.df$clip <- as.factor(acr.sub.total.df$clip)
acr.sub.total.df$dim <- as.factor(acr.sub.total.df$dim)
acr.sub.total.df$tool <- as.factor(acr.sub.total.df$tool)
acr.sub.total.df$id <- c(rep(rep(c(1:90), times = 9), 2), c(1:90)) # indexing the (ref, target) pair
acr.sub.total.df$id <- as.factor(acr.sub.total.df$id)

rm(acr.sub.cebra.df, acr.sub.pca.df, acr.sub.faa.df) # clean the environment

# >> I-1-b. across-clips ####
acr.clip.cebra.df <- read.csv("../results/decoding_performances/across_clip_decoding_CEBRA.csv")
acr.clip.cebra.df$tool <- "CEBRA"
acr.clip.pca.df <- read.csv("../results/decoding_performances/across_clip_decoding_PCA.csv")
acr.clip.pca.df$tool <- "PCA"
acr.clip.faa.df <- read.csv("../results/decoding_performances/across_clip_decoding_FAA.csv")
acr.clip.faa.df$tool <- "FAA"

acr.clip.total.df <- rbind(acr.clip.cebra.df, acr.clip.pca.df)
acr.clip.total.df <- rbind(acr.clip.total.df, acr.clip.faa.df)

acr.clip.total.df$ref_clip <- as.factor(acr.clip.total.df$ref_clip)
acr.clip.total.df$target_clip <- as.factor(acr.clip.total.df$target_clip)
acr.clip.total.df$sub <- as.factor(acr.clip.total.df$sub)
acr.clip.total.df$dim <- as.factor(acr.clip.total.df$dim)
acr.clip.total.df$tool <- as.factor(acr.clip.total.df$tool)
acr.clip.total.df$id <- c(rep(rep(c(1:36), times = 9), 2), c(1:36)) # indexing the (ref, target) pair
acr.clip.total.df$id <- as.factor(acr.clip.total.df$id)

rm(acr.clip.cebra.df, acr.clip.pca.df, acr.clip.faa.df) # clean the environment

## > I-2. Dimensionality Selection ####
# >> I-2-a. across-participants ####
# hierarchical clustering analysis
acr.sub.cebra.clust.df <- acr.sub.total.df[acr.sub.total.df$tool == "CEBRA", ] %>%
  group_by(dim) %>%
  summarize(avg_score = mean(test_f1_align))

acr.sub.cebra.clust.mtx <- dist(acr.sub.cebra.clust.df, method = "euclidean")
acr.sub.cebra.hc <- hclust(acr.sub.cebra.clust.mtx, method = "ward.D2")
acr.sub.cebra.silh <- numeric()

for (k in 2:8) {
  clusters <- cutree(acr.sub.cebra.hc, k = k)
  acr.sub.cebra.silh[k] <- mean(silhouette(clusters, acr.sub.cebra.clust.mtx)[, 3])
}
rm(k, clusters)

acr.sub.cebra.optk <- which.max(acr.sub.cebra.silh)

# visualization (i.e., Fig 4a left panel = fig4a.sub.dend + fig4a.sub.violin)
fig4a.sub.dend <- fviz_dend(acr.sub.cebra.hc,
                            k = acr.sub.cebra.optk, 
                            show_labels = TRUE, rect = FALSE, k_colors = c("#E1BEE7", "#BDBDBD"), 
                            main = "across participants (aligned CEBRA)")

fig4a.sub.violin <- ggplot(acr.sub.total.df[acr.sub.total.df$tool == "CEBRA", ], aes(dim, test_f1_align, fill = dim)) + 
  geom_violin(aes(fill = dim), alpha = 0.75, color = "white") +
  scale_x_discrete(limits = c('9', '6', '7', '8', '4', '5', '1', '2', '3')) + # matching the order of dendrogram
  scale_fill_manual(values = c("#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD", 
                               "#E1BEE7", "#E1BEE7", "#E1BEE7", "#E1BEE7")) +
  geom_point(aes(group = dim), position = position_nudge(0), alpha = 0.25, shape = 16, size = 0.75) +
  geom_line(aes(group = id), position = position_nudge(0), color = "#616161", alpha = 0.05) +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

# >> I-2-b. across-clips ####
# hierarchical clustering analysis
acr.clip.cebra.clust.df <- acr.clip.total.df[acr.clip.total.df$tool == "CEBRA", ] %>%
  group_by(dim) %>%
  summarize(avg_score = mean(test_f1_align))

acr.clip.cebra.clust.mtx <- dist(acr.clip.cebra.clust.df, method = "euclidean")
acr.clip.cebra.hc <- hclust(acr.clip.cebra.clust.mtx, method = "ward.D2")
acr.clip.cebra.silh <- numeric()

for (k in 2:8) {
  clusters <- cutree(acr.clip.cebra.hc, k = k)
  acr.clip.cebra.silh[k] <- mean(silhouette(clusters, acr.clip.cebra.clust.mtx)[, 3])
}
rm(k, clusters)

acr.clip.cebra.optk <- which.max(acr.clip.cebra.silh)

# visualization (i.e., Fig 4a right panel = fig4a.clip.dend + fig4a.clip.violin)
fig4a.clip.dend <- fviz_dend(acr.clip.cebra.hc,
                             k = acr.clip.cebra.optk, 
                             show_labels = TRUE, rect = FALSE, k_colors = c("#E1BEE7", "#BDBDBD"), 
                             main = "across clips (aligned CEBRA)")

fig4a.clip.violin <- ggplot(acr.clip.total.df[acr.clip.total.df$tool == "CEBRA", ], aes(dim, test_f1_align, fill = dim)) + 
  geom_violin(aes(fill = dim), alpha = 0.75, color = "white") +
  scale_x_discrete(limits = c('7', '8', '9', '1', '2', '3', '4', '5', '6')) + # matching the order of dendrogram
  scale_fill_manual(values = c("#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD",
                               "#E1BEE7", "#E1BEE7", "#E1BEE7")) +
  geom_point(aes(group = dim), position = position_nudge(0), alpha = 0.25, shape = 16, size = 0.75) +
  geom_line(aes(group = id), position = position_nudge(0), color = "#616161", alpha = 0.05) +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))


### ________________________________________________________ ####
### @@@ PART II. DIMENSIONALITY SELECTION for PCA [SFIG 2] @@@ ####
## > II-1. Across-Participants ####
# hierarchical clustering analysis
acr.sub.pca.clust.df <- acr.sub.total.df[acr.sub.total.df$tool == "PCA", ] %>%
  group_by(dim) %>%
  summarize(avg_score = mean(test_f1_align))

acr.sub.pca.clust.mtx <- dist(acr.sub.pca.clust.df, method = "euclidean")
acr.sub.pca.hc <- hclust(acr.sub.pca.clust.mtx, method = "ward.D2")
acr.sub.pca.silh <- numeric()

for (k in 2:8) {
  clusters <- cutree(acr.sub.pca.hc, k = k)
  acr.sub.pca.silh[k] <- mean(silhouette(clusters, acr.sub.pca.clust.mtx)[, 3])
}
rm(k, clusters)

acr.sub.pca.optk <- which.max(acr.sub.pca.silh)

# visualization (i.e., SFig 2a  = sfig2a.dend + fig2a.violin)
sfig2a.dend <- fviz_dend(acr.sub.pca.hc,
                         k = acr.sub.pca.optk, 
                         show_labels = TRUE, rect = FALSE, k_colors = c("#E1BEE7", "#BDBDBD"), 
                         main = "across participants (aligned PCA)")

sfig2a.violin <- ggplot(acr.sub.total.df[acr.sub.total.df$tool == "PCA", ], aes(dim, test_f1_align, fill = dim)) + 
  geom_violin(aes(fill = dim), alpha = 0.75, color = "white") +
  scale_x_discrete(limits = c('8', '9', '6', '7', '1', '2', '3', '4', '5')) + # matching the order of dendrogram
  scale_fill_manual(values = c("#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD", 
                               "#E1BEE7", "#E1BEE7", "#E1BEE7", "#E1BEE7")) +
  geom_point(aes(group = dim), position = position_nudge(0), alpha = 0.25, shape = 16, size = 0.75) +
  geom_line(aes(group = id), position = position_nudge(0), color = "#616161", alpha = 0.05) +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

## > II-2. Across-Clips ####
# hierarchical clustering analysis
acr.clip.pca.clust.df <- acr.clip.total.df[acr.clip.total.df$tool == "PCA", ] %>%
  group_by(dim) %>%
  summarize(avg_score = mean(test_f1_align))

acr.clip.pca.clust.mtx <- dist(acr.clip.pca.clust.df, method = "euclidean")
acr.clip.pca.hc <- hclust(acr.clip.pca.clust.mtx, method = "ward.D2")
acr.clip.pca.silh <- numeric()

for (k in 2:8) {
  clusters <- cutree(acr.clip.pca.hc, k = k)
  acr.clip.pca.silh[k] <- mean(silhouette(clusters, acr.clip.pca.clust.mtx)[, 3])
}
rm(k, clusters)

acr.clip.pca.optk <- which.max(acr.clip.pca.silh)

# visualization (i.e., SFig 2b = sfig2b.dend + sfig2b.violin)
sfig2b.dend <- fviz_dend(acr.clip.pca.hc,
                         k = acr.clip.pca.optk, 
                         show_labels = TRUE, rect = FALSE, k_colors = c("#BDBDBD", "#E1BEE7"), 
                         main = "across clips (aligned PCA)")

sfig2b.violin <- ggplot(acr.clip.total.df[acr.clip.total.df$tool == "PCA", ], aes(dim, test_f1_align, fill = dim)) + 
  geom_violin(aes(fill = dim), alpha = 0.75, color = "white") +
  scale_x_discrete(limits = c('3', '4', '1', '2', '8', '9', '5', '6', '7')) + # matching the order of dendrogram
  scale_fill_manual(values = c("#BDBDBD", "#BDBDBD", "#BDBDBD", "#BDBDBD",
                               "#E1BEE7", "#E1BEE7", "#E1BEE7", "#E1BEE7", "#E1BEE7")) +
  geom_point(aes(group = dim), position = position_nudge(0), alpha = 0.25, shape = 16, size = 0.75) +
  geom_line(aes(group = id), position = position_nudge(0), color = "#616161", alpha = 0.05) +
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))


### ________________________________________________________ ####
### @@@ PART III. STATISTICAL ANALYSES FOR DECODING [FIG 4B & STABLE 4] @@@ ####
## > III-1. across-participants ####
# >> III-1-a. descriptive stats (Fig 4b) ####
stat.sub.cebra.selected <- acr.sub.total.df[(acr.sub.total.df$dim == 7) & (acr.sub.total.df$tool == "CEBRA"), 
                                           c("ref_sub", "target_sub", "clip", "tool", 
                                             "test_f1_random", "test_f1_not_align", "test_f1_align")]

stat.sub.pca.selected <- acr.sub.total.df[(acr.sub.total.df$dim == 6) & (acr.sub.total.df$tool == "PCA"), 
                                          c("ref_sub", "target_sub", "clip", "tool",
                                            "test_f1_random", "test_f1_not_align", "test_f1_align")]

stat.sub.faa.selected <- acr.sub.total.df[(acr.sub.total.df$dim == 1) & (acr.sub.total.df$tool == "FAA"), 
                                          c("ref_sub", "target_sub", "clip", "tool",
                                            "test_f1_random", "test_f1_not_align", "test_f1_align")]

stat.sub.selected <- rbind(stat.sub.cebra.selected, stat.sub.pca.selected)
stat.sub.selected <- rbind(stat.sub.selected, stat.sub.faa.selected)

rm(stat.sub.cebra.selected, stat.sub.pca.selected, stat.sub.faa.selected) # clean the environment

stat.sub.selected <- stat.sub.selected %>%
  pivot_longer(
    cols = starts_with("test_f1_"),  
    names_to = "condition",        
    values_to = "test_f1"            
  )

stat.sub.selected$id <- rep(rep(c(1:90), each = 3), 3) # matching (ref, target) pairs
stat.sub.selected$id <- as.factor(stat.sub.selected$id)
stat.sub.selected$condition <- as.factor(stat.sub.selected$condition)

# line plot
stat.sub.stats.line <- stat.sub.selected %>%
  group_by(condition, tool) %>%
  summarise(
    sd = sd(test_f1),
    sem = sd(test_f1)/sqrt(90),
    mean = mean(test_f1)
  )

fig4b.sub.line <- ggplot(stat.sub.stats.line, aes(condition, mean)) +
  geom_errorbar(
    aes(ymin = mean - 1.96 * sem, ymax = mean + 1.96 * sem, color = tool),
    position = position_dodge(0.55), width = 0
  ) +
  geom_point(aes(color = tool), position = position_dodge(0.55)) +
  geom_line(
    aes(group = tool, color = tool, linetype = tool), 
    position = position_dodge(0.55)
  ) +
  scale_color_manual(values = c("#BA68C8", "#BDBDBD", "#757575")) +
  scale_linetype_manual(values = c("CEBRA" = "solid", "PCA" = "dashed", "FAA" = "dotted")) +
  scale_x_discrete(limits = c("test_f1_random", "test_f1_not_align", "test_f1_align")) +
  theme_bw() +
  ylim(c(0.0, 0.5)) + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

# heatmap
stat.sub.stats.heat <- stat.sub.selected[stat.sub.selected$tool == "CEBRA" & 
                                           stat.sub.selected$condition == "test_f1_align", ] %>%
  group_by(ref_sub, target_sub) %>%
  summarise(mean = mean(test_f1, na.rm = TRUE)) %>%
  ungroup()

fig4b.sub.heat <- ggplot(stat.sub.stats.heat, aes(x = ref_sub, y = target_sub, fill = mean)) +
  geom_tile(color = "white", linewidth = 0.75) +
  scale_fill_gradientn(colors = c("white", "#BA68C8"), 
                       name = "test F1 score",
                       limits = c(0, 1)) +
  labs(title = "Across Participants Decoding Performance (Aligned CEBRA)",
       x = "train participant",
       y = "test participant") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank())

# >> III-1-b. statistical test (STable 4) ####
extract_posthoc_summary <- function(posthoc_result, model) {
  df <- as.data.frame(posthoc_result$contrasts)
  
  # Residual SD from the model
  resid_sd <- sigma(model)
  
  # Cohen's d = estimate / residual SD
  df$cohens_d <- df$estimate / resid_sd
  
  # Rename for clarity
  df_out <- data.frame(
    contrast = df$contrast,
    t = round(as.numeric(df$t.ratio), 3),
    df = as.numeric(df$df),
    CI_lower = round(as.numeric(df$estimate - 1.96*df$SE), 3),
    CI_upper = round(as.numeric(df$estimate + 1.96*df$SE), 3),
    cohens_d = round(as.numeric(df$cohens_d), 3),
    p_FDR = signif(as.numeric(df$p.value), 3)
  )
  
  return(df_out)
}


# >>> III-1-b-(1). within CEBRA ####
sub_condition.model <- lmer(test_f1 ~ condition + (1 | id), data = stat.sub.selected)
sub_condition.posthoc <- emmeans(sub_condition.model, pairwise ~ condition, adjust = "fdr")
sub_condition.df <- extract_posthoc_summary(sub_condition.posthoc, sub_condition.model)

# >>> III-1-b-(2). between tools ####
sub_tool_random.model <- lmer(test_f1 ~ tool + (1 | id), data = stat.sub.selected[stat.sub.selected$condition == 'test_f1_random', ])
sub_tool_random.posthoc <- emmeans(sub_tool_random.model, pairwise ~ tool, adjust = "fdr")
sub_tool_random.df <- extract_posthoc_summary(sub_tool_random.posthoc, sub_tool_random.model)

sub_tool_notaligned.model <- lmer(test_f1 ~ tool + (1 | id), data = stat.sub.selected[stat.sub.selected$condition == 'test_f1_not_align', ])
sub_tool_notaligned.posthoc <- emmeans(sub_tool_notaligned.model, pairwise ~ tool, adjust = "fdr")
sub_tool_notaligned.df <- extract_posthoc_summary(sub_tool_notaligned.posthoc, sub_tool_notaligned.model)

sub_tool_aligned.model <- lmer(test_f1 ~ tool + (1 | id), data = stat.sub.selected[stat.sub.selected$condition == 'test_f1_align', ])
sub_tool_aligned.posthoc <- emmeans(sub_tool_aligned.model, pairwise ~ tool, adjust = "fdr")
sub_tool_aligned.df <- extract_posthoc_summary(sub_tool_aligned.posthoc, sub_tool_aligned.model)

decoding.stat.across_sub <- rbind(
  cbind(context = "within_condition", sub_condition.df),
  cbind(context = "between_tool_random", sub_tool_random.df),
  cbind(context = "between_tool_notaligned", sub_tool_notaligned.df),
  cbind(context = "between_tool_aligned", sub_tool_aligned.df)
) # STable 4


## > III-2. across-clips ####
# >> III-2-a. descriptive stats (Fig 4b) ####
stat.clip.cebra.selected <- acr.clip.total.df[(acr.clip.total.df$dim == 7) & (acr.clip.total.df$tool == "CEBRA"), 
                                              c("ref_clip", "target_clip", "sub", "tool", 
                                                "test_f1_random", "test_f1_not_align", "test_f1_align")]

stat.clip.pca.selected <- acr.clip.total.df[(acr.clip.total.df$dim == 6) & (acr.clip.total.df$tool == "PCA"), 
                                            c("ref_clip", "target_clip", "sub", "tool",
                                              "test_f1_random", "test_f1_not_align", "test_f1_align")]

stat.clip.faa.selected <- acr.clip.total.df[(acr.clip.total.df$dim == 1) & (acr.clip.total.df$tool == "FAA"), 
                                            c("ref_clip", "target_clip", "sub", "tool",
                                              "test_f1_random", "test_f1_not_align", "test_f1_align")]

stat.clip.selected <- rbind(stat.clip.cebra.selected, stat.clip.pca.selected)
stat.clip.selected <- rbind(stat.clip.selected, stat.clip.faa.selected)

rm(stat.clip.cebra.selected, stat.clip.pca.selected, stat.clip.faa.selected) # clean the environment

stat.clip.selected <- stat.clip.selected %>%
  pivot_longer(
    cols = starts_with("test_f1_"),  
    names_to = "condition",        
    values_to = "test_f1"            
  )

stat.clip.selected$id <- rep(rep(c(1:36), each = 3), 3) # matching (ref, target) pairs
stat.clip.selected$id <- as.factor(stat.clip.selected$id)
stat.clip.selected$condition <- as.factor(stat.clip.selected$condition)

# line plot
stat.clip.stats.line <- stat.clip.selected %>%
  group_by(condition, tool) %>%
  summarise(
    sd = sd(test_f1),
    sem = sd(test_f1)/sqrt(36),
    mean = mean(test_f1)
  )

fig4b.clip.line <- ggplot(stat.clip.stats.line, aes(condition, mean)) +
  geom_errorbar(
    aes(ymin = mean - 1.96 * sem, ymax = mean + 1.96 * sem, color = tool),
    position = position_dodge(0.55), width = 0
  ) +
  geom_point(aes(color = tool), position = position_dodge(0.55)) +
  geom_line(
    aes(group = tool, color = tool, linetype = tool), 
    position = position_dodge(0.55)
  ) +
  scale_color_manual(values = c("#BA68C8", "#BDBDBD", "#757575")) +
  scale_linetype_manual(values = c("CEBRA" = "solid", "PCA" = "dashed", "FAA" = "dotted")) +
  scale_x_discrete(limits = c("test_f1_random", "test_f1_not_align", "test_f1_align")) +
  ylim(c(0.0, 0.5)) + 
  theme_bw() +
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

# heatmap
stat.clip.stats.heat <- stat.clip.selected[stat.clip.selected$tool == "CEBRA" & 
                                             stat.clip.selected$condition == "test_f1_align", ] %>%
  group_by(ref_clip, target_clip) %>%
  summarise(mean = mean(test_f1, na.rm = TRUE)) %>%
  ungroup()

fig4b.clip.heat <- ggplot(stat.clip.stats.heat, aes(x = ref_clip, y = target_clip, fill = mean)) +
  geom_tile(color = "white", linewidth = 0.75) +
  scale_fill_gradientn(colors = c("white", "#BA68C8"), 
                       name = "test F1 score",
                       limits = c(0, 1)) +
  labs(title = "Across Clips Decoding Performance (Aligned CEBRA)",
       x = "train clip",
       y = "test clip") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1),
        panel.grid = element_blank())


# >> III-2-b. statistical test (STable 4) ####
# >>> III-2-b-(1). within CEBRA ####
clip_condition.model <- lmer(test_f1 ~ condition + (1 | id), data = stat.clip.selected)
clip_condition.posthoc <- emmeans(clip_condition.model, pairwise ~ condition, adjust = "fdr")
clip_condition.df <- extract_posthoc_summary(clip_condition.posthoc, clip_condition.model)

# >>> III-2-b-(2). between tools ####
clip_tool_random.model <- lmer(test_f1 ~ tool + (1 | id), data = stat.clip.selected[stat.clip.selected$condition == 'test_f1_random', ])
clip_tool_random.posthoc <- emmeans(clip_tool_random.model, pairwise ~ tool, adjust = "fdr")
clip_tool_random.df <- extract_posthoc_summary(clip_tool_random.posthoc, clip_tool_random.model)

clip_tool_notaligned.model <- lmer(test_f1 ~ tool + (1 | id), data = stat.clip.selected[stat.clip.selected$condition == 'test_f1_not_align', ])
clip_tool_notaligned.posthoc <- emmeans(clip_tool_notaligned.model, pairwise ~ tool, adjust = "fdr")
clip_tool_notaligned.df <- extract_posthoc_summary(clip_tool_notaligned.posthoc, clip_tool_notaligned.model)

clip_tool_aligned.model <- lmer(test_f1 ~ tool + (1 | id), data = stat.clip.selected[stat.clip.selected$condition == 'test_f1_align', ])
clip_tool_aligned.posthoc <- emmeans(clip_tool_aligned.model, pairwise ~ tool, adjust = "fdr")
clip_tool_aligned.df <- extract_posthoc_summary(clip_tool_aligned.posthoc, clip_tool_aligned.model)

decoding.stat.across_clip <- rbind(
  cbind(context = "within_condition", clip_condition.df),
  cbind(context = "between_tool_random", clip_tool_random.df),
  cbind(context = "between_tool_notaligned", clip_tool_notaligned.df),
  cbind(context = "between_tool_aligned", clip_tool_aligned.df)
)

# STable 4
write.csv(decoding.stat.across_sub, "../results/decoding_performances/decoding_stat_across_sub.csv")
write.csv(decoding.stat.across_clip, "../results/decoding_performances/decoding_stat_across_clip.csv")

### ________________________________________________________ ####
### RESULTS ####
fig4a.sub.dend
fig4a.sub.violin

fig4a.clip.dend
fig4a.clip.violin

fig4b.sub.line
fig4b.sub.heat

fig4b.clip.line
fig4b.clip.heat

sfig2a.dend
sfig2a.violin

sfig2b.dend
sfig2b.violin