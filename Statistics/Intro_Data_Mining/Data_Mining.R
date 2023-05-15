# Midterm Project of Data Mining 

# Preparation
#install.packages('dslabs')
#install.packages('ggplot2')
#install.packages('tidyr')
#install.packages('caret')
#install.packages('car')
#install.packages('class')
library(caret)

path1 <- "Data_Mining/Datasets/breast-cancer.txt"
path2 <- "Data_Mining/Datasets/Teaching Assistant Evaluation.txt"
path3 <- "Data_Mining/Datasets/Tic-Tac-Toe Endgame.txt"

bc_dataset <- read.table(path1, sep = ',')
tae_dataset <- read.table(path2, sep = ',')
ttt_dataset <- read.table(path3, sep = ',')

colnames(bc_dataset) <- c('Class', 'age', 'menopause', 'tumor-size', 
                          'inv-nodes', 'node-caps', 'deg-malig', 'breast',
                          'breast-quad', 'irradate')

colnames(tae_dataset) <- c('native_speaker', 'course_instructor', 'course', 'summer_regular', 
                           'Class_size', 'Class_attribute')

colnames(ttt_dataset) <- c('top_left', 'top_middle', 'top_right', 'middle_left',
                           'middle_middle', 'middle_right', 'bottom_left', 
                           'bottom_middle', 'bottom_right', 'Class')

# 1. breast-cancer

# Check dataset
head(bc_dataset)
str(bc_dataset)
dim(bc_dataset)
summary(bc_dataset)

# Preprocessing 

# Zero Variance test
nzv <- nearZeroVar(bc_dataset, saveMetrics = TRUE) 
nzv

# Logistic Regression 

train_idx <- sample(1:nrow(bc_dataset), size = 0.8*nrow(bc_dataset), replace = FALSE)
test_idx <- (-train_idx)

bc_train <- bc_dataset[train_idx, ]
bc_test <- bc_dataset[test_idx, ]

model <- glm(Class~., data = bc_train, family = 'binomial')
step_model <- step(model, direction = 'both')

summary(step_model)

null_deviance <- 277.87
residual_deviance <- 245.89
model_deviance <- null_deviance - residual_deviance 
pchisq(model_deviance, df = 2, lower.tail = FALSE)

# Testing VIF
library(car)
vif(step_model)

# Make Prediction

pred <- predict(step_model, newdata = bc_test[, -1], type = 'response')
df_pred <- as.data.frame(pred)

df_pred$default <- ifelse(df_pred$pred > 0.5, df_pred$default <- "recurrence-events", df_pred$default <- "no-recurrence-events")
df_pred$default <- as.factor(df_pred$default)

# Model Evaluation 
caret::confusionMatrix(df_pred$default, bc_test[, 'Class'])
library(ModelMetrics)
auc(actual = bc_test[,'Class'], predicted = df_pred$default)


# 2. Teaching Assistant Evaluation
head(tae_dataset)
str(tae_dataset)
dim(tae_dataset)
summary(tae_dataset)

# Preprocessing 
# Zero Variance test
nzv <- nearZeroVar(tae_dataset, saveMetrics = TRUE)
nzv

# Logistic Regression 

train_idx <- sample(1:nrow(tae_dataset), size = 0.8*nrow(tae_dataset), replace = FALSE)
test_idx <- (-train_idx)

tae_train <- tae_dataset[train_idx, ]
tae_test <- tae_dataset[test_idx, ]

model <- nnet::multinom(Class_attribute~., data = tae_train)
step_model <- step(model, direction = 'both')

summary(step_model)

# Make Prediction

pred <- predict(step_model, newdata = tae_test[, -6], type = 'class')
df_pred <- as.data.frame(pred)

# Model Evaluation 
caret::confusionMatrix(df_pred$pred, as.factor(tae_test[, 'Class_attribute']))

install.packages('randomForest')
# RandomForest
library(ran)

# 3. Tic-Tac-Toe Endgame
head(ttt_dataset)
str(ttt_dataset)
dim(ttt_dataset)
summary(ttt_dataset)

# Decision Tree 

train_idx <- sample(1:nrow(ttt_dataset), size = 0.8*nrow(ttt_dataset), replace = FALSE)
test_idx <- (-train_idx)

train_X <- ttt_dataset[train_idx, ]
test_X <- ttt_dataset[test_idx, ]

library(rpart)
model <- rpart(Class ~., data = train_X)
model

plot(model, compress = TRUE, margin = 0.5)
text(model, cex = 0.5)

library(rpart.plot)
prp(model, type = 2, extra = 2)

# Model Evaluation

model$cptable
plotcp(model)

tree_pred <- predict(model, newdata = test_X, type = 'class')
confusionMatrix(tree_pred, reference = test_X$Class)
