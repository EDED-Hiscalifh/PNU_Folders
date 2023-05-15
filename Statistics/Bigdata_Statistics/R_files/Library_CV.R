# 1. All packages needed for Bigdata Statistics 

library(class)
library(tree)
library(randomForest)
library(glmnet)
library(glmnet)
library(MASS)
library(leaps)
library(ISLR)
library(nnet)
library(MASS)
library(e1071)
library(glmnet)
library(class)
library(tree)
library(randomForest)
library(gbm)
library(JOUSBoost)

# 2. Validation Set Approach 

# Dataset Preparation 
library(ISLR) 
data(Auto) 
str(Auto) 
summary(Auto) 

# Extract target 
mpg <- Auto$mpg
horsepower <- Auto$horsepower

# set df 
dg <- 1:9
u <- order(horsepower) 

# Single Split 
set.seed(1)
n <- nrow(Auto)

## training set
tran <- sample(n, n/2)
MSE <- NULL
for (k in 1:length(dg)) {
  g <- lm(mpg ~ poly(horsepower, dg[k]), subset=tran)
  MSE[k] <- mean((mpg - predict(g, Auto))[-tran]^2)
}

# Visualization MSE_test
plot(dg, MSE, type="b", col=2, xlab="Degree of Polynomial",
     ylab="Mean Squared Error", ylim=c(15,30), lwd=2, pch=19)
abline(v=which.min(MSE), lty=2)

# 3. K-fold Cross Validation 

# 10-fold cross validation
K <- 10 
MSE <- matrix(0, n, length(dg)) # degree is 1:9

# Assertion each data point to each fold 
# e.g. [1, 3, 3, 5, 6, ..., 10] (n) 
set.seed(1234) 
u <- sample(rep(seq(K), length=n)) 

# Model training 

for (k in 1:K) {
  tran <- which(u!=k) 
  test <- which(u==k) 
  for (i in 1:length(dg)) { 
    g <- lm(mpg ~ poly(horsepower, i), subset=tran) 
    MSE[test, i] <- (mpg - predict(g, Auto))[test]^2 
  } 
}
CVE <- apply(MSE, 2, mean) 

# Visualization
plot(dg, CVE, type="b", col="darkblue",
     xlab="Degree of Polynomial", ylab="Mean Squared Error",
     ylim=c(18,25), lwd=2, pch=19)
abline(v=which.min(CVE), lty=2)

# 4. LOOCV 

# Auto Data : LOOCV
# Set the degree of freedom and result matrix 
n <- nrow(Auto)
dg <- 1:9
MSE <- matrix(0, n, length(dg))

for (i in 1:n) {
  for (k in 1:length(dg)) {
    g <- lm(mpg ~ poly(horsepower, k), subset=(1:n)[-i])
    MSE[i, k] <- mean((mpg - predict(g, Auto))[i]^2)
  }
}
# Calculate CVE 
aMSE <- apply(MSE, 2, mean)

# Visualization
plot(dg, aMSE, type="b", col="darkblue",
     xlab="Degree of Polynomial", ylab="Mean Squared Error",
     ylim=c(18,25), lwd=2, pch=19)
abline(v=which.min(aMSE), lty=2)