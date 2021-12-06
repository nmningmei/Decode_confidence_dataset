# first set the working directory to the source file location
working_dir <- "../results"
working_data <- Sys.glob(file.path(working_dir,"for_anova","*csv"))
saving_dir <- "repeated_measure_anova"
library(lmerTest)

ii <- 1
df <- read.csv(working_data[ii])
saving_name <- strsplit(working_data[ii],"/")[[1]][4]
df$decoder <- as.factor(df$decoder)
df$condition <- as.factor(df$condition)
df$domain <- as.factor(df$source)
df$filename <- as.factor(df$filename)
fit <- lmer(score~decoder*domain*condition+(1|filename),data = df)
res <- anova(fit)
capture.output(as.data.frame(res),file = file.path(working_dir,saving_dir,saving_name))

ii <- 2
df <- read.csv(working_data[ii])
saving_name <- strsplit(working_data[ii],"/")[[1]][4]
df$decoder <- as.factor(df$decoder)
df$condition <- as.factor(df$condition)
df$study_name <-as.factor(df$study_name)
fit <- lmer(score~decoder*condition+(1|study_name),data = df)
res <- anova(fit)
capture.output(as.data.frame(res),file = file.path(working_dir,saving_dir,saving_name))

ii <- 3
df <- read.csv(working_data[ii])
saving_name <- strsplit(working_data[ii],"/")[[1]][4]
df$decoder <- as.factor(df$decoder)
df$condition <- as.factor(df$condition)
df$domain <- as.factor(df$source)
df$accuracy_train <- df$accuracy_train
df$accuracy_test <- df$accuracy_test
df$filename <- as.factor(df$filename)
fit <- lmer(score~decoder*condition*domain*accuracy_train*accuracy_test+(1|filename),data = df)
res <- anova(fit)
capture.output(as.data.frame(res),file = file.path(working_dir,saving_dir,saving_name))

ii <- 4
df <- read.csv(working_data[ii])
saving_name <- strsplit(working_data[ii],"/")[[1]][4]
df$decoder <- as.factor(df$decoder)
df$condition <- as.factor(df$condition)
df$accuracy_train <- df$accuracy_train
df$accuracy_test <- df$accuracy_test
df$study_name <- as.factor(df$study_name)
fit <- lmer(score~decoder*condition*accuracy_train*accuracy_test+(1|study_name),data = df)
res <- anova(fit)
capture.output(as.data.frame(res),file = file.path(working_dir,saving_dir,saving_name))