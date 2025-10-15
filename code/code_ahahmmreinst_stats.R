# R 4.4.2

# install.packages("lme4")
# install.packages("Matrix")
# install.packages("lmerTest")
# install.packages("mediation")
library(lmerTest)
library(lme4)
library(Matrix)
library(mediation)
data <- read.csv("/data/summarydata_ahahmmreinst.csv")
df <- data.frame(
  subject = as.factor(data$subject),
  parcel = as.factor(data$parcel),
  reinst = rowMeans(data[, c("reinstatement.m7TR", "reinstatement.m6TR", 
                             "reinstatement.m5TR", "reinstatement.m4TR", 
                             "reinstatement.m3TR")], na.rm = TRUE),
  shift2 = as.integer(data$shift.m2TR),
  shift1 = as.integer(data$shift.m1TR),
  retrieval = as.integer(data$retrieval)
)
df$shift12 <- as.integer(df$shift1+df$shift2)
df <- na.omit(df)

# Figure 4B
model <- glmer(retrieval ~ reinst + parcel + (1 | subject), data = df, family = binomial)
summary(model)
model_reduced_rein <- glmer(retrieval ~ parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_rein, model)

model <- glmer(shift2 ~ reinst + parcel + (1 | subject), data = df, family = binomial)
summary(model)
model_reduced_rein <- glmer(shift2 ~ parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_rein, model)


# Behavioral retrieval ~ Neural reinstatement + Neural pattern shift
model <- glmer(retrieval ~ shift2 + reinst + shift2:reinst + parcel + (1 | subject), data = df, family = binomial)
summary(model)

model_reduced_rein <- glmer(retrieval ~ shift2 + shift2:reinst + parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_rein, model)

model_reduced_hmm <- glmer(retrieval ~ reinst + shift2:reinst + parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_hmm, model)

model_reduced_inter <- glmer(retrieval ~ shift2 + reinst + parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_inter, model)

# Behavioral retrieval ~ Neural reinstatement + Neural pattern shift (-2 TR and -1 TR)
# swapping "shift2" variable to "shift12"
