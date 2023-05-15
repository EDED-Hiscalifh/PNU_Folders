# Import library 
library(ISLR) 
data(Default) 
attach(Default) 
library(MASS) 

# Train-test split
set.seed(1234)
n <- length(default) 
train <- sample(1:n, n*0.7) 
test <- setdiff(1:n, train) 

# Classification error rate of LDA 
g1 <- lda(default~., data=Default, subset=train)
pred1 <- predict(g1, Default) 
table(pred1$class[test], Default$default[test]) 
mean(pred1$class[test]!=Default$default[test])

# Classification error rate of QDA
g2 <- qda(default~., data=Default, subset=train)
pred2 <- predict(g2, Default)
table(pred2$class[test], Default$default[test])
mean(pred2$class[test]!=Default$default[test])

# AUC comparison between LDA and QDA
library(ROCR)
label <- factor(default[test], levels=c("Yes","No"),
                labels=c("TRUE","FALSE"))
preds1 <- prediction(pred1$posterior[test,2], label)
preds2 <- prediction(pred2$posterior[test,2], label)
performance(preds1, "auc")@y.values
performance(preds2, "auc")@y.values

# Simulation Study 
set.seed(123)
N <- 100
CER <- AUC <- matrix(NA, N, 2)
for (i in 1:N) {
  train <- sample(1:n, n*0.7)
  test <- setdiff(1:n, train)
  y.test <- Default$default[test]
  50 / 80
  g1 <- lda(default~., data=Default, subset=train)
  g2 <- qda(default~., data=Default, subset=train)
  pred1 <- predict(g1, Default)
  pred2 <- predict(g2, Default)
  CER[i,1] <- mean(pred1$class[test]!=y.test)
  CER[i,2] <- mean(pred2$class[test]!=y.test)
  label <- factor(default[test], levels=c("Yes","No"),
                  labels=c("TRUE","FALSE"))
  preds1 <- prediction(pred1$posterior[test,2], label)
  preds2 <- prediction(pred2$posterior[test,2], label)
  AUC[i,1] <- as.numeric(performance(preds1, "auc")@y.values)
  AUC[i,2] <- as.numeric(performance(preds2, "auc")@y.values)
}
apply(CER, 2, mean)
apply(AUC, 2, mean)