# install.packages("lme4")
# install.packages("Matrix")
# install.packages("lmerTest")
library(lme4)
library(Matrix)
library(lmerTest)
data <- read.csv("/Users/hayoungsong/Documents/2024COCO/github/data/summarydata_ahahmmreinst.csv")
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


# Model 1
model <- glmer(shift2 ~ reinst + parcel + (1 | subject), data = df, family = binomial)
summary(model)
model_reduced_rein <- glmer(shift2 ~ parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_rein, model)

# Model 2
model <- glmer(retrieval ~ shift2 + reinst + shift2:reinst + parcel + (1 | subject), data = df, family = binomial)
summary(model)

model_reduced_rein <- glmer(retrieval ~ shift2 + shift2:reinst + parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_rein, model)

model_reduced_hmm <- glmer(retrieval ~ reinst + shift2:reinst + parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_hmm, model)

model_reduced_inter <- glmer(retrieval ~ shift2 + reinst + parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_inter, model)

# Model 3
model <- glmer(retrieval ~ shift12 + reinst + shift12:reinst + parcel + (1 | subject), data = df, family = binomial)
summary(model)
model_reduced_hmm <- glmer(retrieval ~ reinst + shift12:reinst + parcel + (1 | subject), data = df, family = binomial)
anova(model_reduced_hmm, model)

