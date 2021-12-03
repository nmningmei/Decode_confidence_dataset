library(lmerTest)
saving_dir <- "repeated_measure_anova"
working_dir <- "../results"

# pair-comparisons of scores
df <- read.csv(file.path(working_dir,"for_anova","scores_LOO.csv"))
df$decoder <- as.factor(df$decoder)
df$accuracy_train <- as.factor(df$accuracy_train)
df$accuracy_test <- as.factor(df$accuracy_test)
df$study_name <- as.factor(df$study_name)
fit <- aov(score~decoder*accuracy_train*accuracy_test+Error(study_name/(decoder*accuracy_train*accuracy_test)),data = df)
res <- summary(fit)
capture.output(res,file = file.path(working_dir,saving_dir,'scores_LOO_rm_anova.txt'))

df <- read.csv(file.path(working_dir,"for_anova","scores_cross_domain.csv"))
df$decoder <- as.factor(df$decoder)
df$accuracy_train <- as.factor(df$accuracy_train)
df$accuracy_test <- as.factor(df$accuracy_test)
df$source <- as.factor(df$source)
df$fold <- as.factor(df$fold)
fit <- aov(score~decoder*accuracy_train*accuracy_test*source+Error(fold/(decoder*accuracy_train*accuracy_test*source)),data = df)
res <- summary(fit)
capture.output(res,file = file.path(working_dir,saving_dir,'scores_cross_domain_rm_anova.txt'))