# ____________________________________________________________________________________________
# AWE IS CHARACTERIZED AS AN AMBIVALENT AFFECT IN THE HUMAN BEHAVIOR AND CORTEX
# _________________________________
# 11 sensory_affect_analysis_n43.R
# _________________________________
# written by
# /
# Jinwoo Lee (SNU Connectome Lab)
# e-mail:  adem1997@snu.ac.kr
# website: jinwoo-lee.com
# _________________________________
# July, 2025
# ____________________________________________________________________________________________
# [MATERIALS CREATED BY THIS SCRIPT] STable 2, SFig 1b

library(ggplot2)
library(mclogit)
library(caret)

### ________________________________________________________ ####
### @@@ PART I. DATA PREPARATION @@@ ####
## > I-1. Data Loading ####
sub.list.behav <- read.csv("../dataset/sub-list-full_n43.csv", header = FALSE)
sub.list.behav <- sub.list.behav[[1]]
clip.list <- c("SP", "CI", "MO", "PA")
time.len <- 233

emo.path <- "../dataset/eeg/pped/"      
sense.path <- "../dataset/movie/"  

total.df <- data.frame(matrix(nrow = 0, ncol = 8))

# save the valence and sensory information
for (sub in sub.list.behav) {
  for (clip in clip.list) {
    
    # loading current sub-clip's valence sequence
    current.emo.name <- paste0(emo.path, sub, "/CEBRA_input/", sub, "_", clip, "-STFT-CEBRA.csv")
    current.emo <- read.csv(current.emo.name)[, "mode"]
    
    # loading current clip's sensory features
    current.clip.bright <- unlist(read.csv(paste0(sense.path, clip, "_bright.csv")), use.name = FALSE)
    current.clip.hue <- unlist(read.csv(paste0(sense.path, clip, "_hue.csv")), use.name = FALSE)
    current.clip.loud <- unlist(read.csv(paste0(sense.path, clip, "_loudness.csv")), use.name = FALSE)
    
    for (t in c(1:time.len)) {
      total.df <- rbind(total.df, c(t, 
                                    current.clip.bright[t], current.clip.hue[t],
                                    current.clip.loud[t], current.emo[t], sub, clip))
    }
    
    print(paste0("Valence and sensory vectors of ", sub, "'s ", clip, " data were successfully saved!"))
  }
}

rm(sub, clip, t, current.emo.name, current.emo, current.clip.bright, current.clip.hue, current.clip.loud)

colnames(total.df) <- c("time", "bright", "hue", "loud", "valence", "subjectkey", "clip")

total.df$time <- as.numeric(total.df$time)
total.df$bright <- as.numeric(total.df$bright)
total.df$hue <- as.numeric(total.df$hue)
total.df$loud <- as.numeric(total.df$loud)
total.df$valence <- as.factor(total.df$valence)
total.df$subjectkey <- as.factor(total.df$subjectkey)
total.df$clip <- as.factor(total.df$clip)

## > I-2. Data Preprocessing ####
std.model <- preProcess(total.df, method = c("center", "scale"))
total.df.std <- predict(std.model, total.df)

norm.model <- preProcess(total.df.std, method = c("range"))
total.df.std.norm <- predict(norm.model, total.df.std)


### ________________________________________________________ ####
### @@@ PART II. MULTINOMIAL MIXED LOGISTIC REGRESSION (STABLE 2) @@@ ####
model <- mblogit(formula = valence ~ time + bright + hue + loud, 
                 random = list(~1|subjectkey, ~1|clip),
                 data = total.df.std.norm)

summary(model)


### ________________________________________________________ ####
### @@@ PART III. VISUALIZATION OF SENSORY DYNAMICS (SFIG 1B) @@@ ####
awe.clips <- c("SP", "CI", "MO")
fig.df <- total.df[total.df$subjectkey == 'sub-01' & total.df$clip %in% awe.clips, ]

## > III-1. Brightness ####
sfig1b.bright <- ggplot() + 
  geom_line(mapping = aes(x = time, y = bright, color = clip), data = fig.df) +
  scale_color_manual(values = c("SP" = "#212121",
                                "CI" = "#9E9E9E", 
                                "MO" = "#E0E0E0")) +  
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

## > III-2. Color Hue ####
sfig1b.hue <- ggplot() + 
  geom_line(mapping = aes(x = time, y = hue, color = clip), data = fig.df) +
  scale_color_manual(values = c("SP" = "#212121",
                                "CI" = "#9E9E9E", 
                                "MO" = "#E0E0E0")) +  
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

## > III-3. Loudness ####
sfig1b.loud <- ggplot() + 
  geom_line(mapping = aes(x = time, y = loud, color = clip), data = fig.df) +
  scale_color_manual(values = c("SP" = "#212121",
                                "CI" = "#9E9E9E", 
                                "MO" = "#E0E0E0")) +  
  theme_bw() + 
  theme(panel.border = element_blank(), panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(), axis.line = element_line(colour = "black"))

### ________________________________________________________ ####
### RESULTS ####
sfig1b.bright
sfig1b.hue
sfig1b.loud
