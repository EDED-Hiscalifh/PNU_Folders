# ======================= Introduction =========================

library(mvtnorm)
set.seed(112233)
K <- 100; n <- 200; p <- 20
x.tran <- x.test <- x.vald <- array(0, c(n, p, K))
z <- rep(c(1, 2, 3), each=n/2)
for (i in 1:K) {
  c <- runif(1, 0, 0.3)
  cov <- matrix(c, p, p); diag(cov) <- 1
  t <- sample(1:p, 1); s <- sample(1:p, t) 
  mu <- rep(0, p); mu[s] <- runif(t, -1, 1) 
  x1 <- rmvt(3*n/2, delta=mu, sigma=diag(p), df=9)
  x2 <- rmvt(3*n/2, delta=rep(0, p), sigma=cov, df=9)
  x.tran[,,i] <- rbind(x1[z==1,], x2[z==1,])
  x.test[,,i] <- rbind(x1[z==2,], x2[z==2,])
  x.vald[,,i] <- rbind(x1[z==3,], x2[z==3,])
}

y <- as.factor(c(rep(1, n/2), rep(-1, n/2)))

# Understanding generated sample sets 
# x.tran : [[v1.tran, v2.tran, ..., v20.tran] * 200] * 100 (200 x 20) x 100 
# x.test : [[v1.test, v2.test, ..., v20.test] * 200] * 100 (200 x 20) x 100 
# x.vald : [[v1.vald, v2.vald, ..., v20.vald] * 200] * 100 (200 x 20) x 100 

# y : ['1' * 100, '-1' * 100] (200 x 1)

library(ISLR)
library(nnet)
library(MASS)
library(e1071) 
library(glmnet) 
library(class)
library(tree)
library(randomForest)
library(gbm)

# ======================== Problem 04 ==========================
# What need to apply 
# 1. Fit a boosting model with the training set where interaction.depth is 1 - 4.
# 2. Find the optimal number of interaction.depth that has the smallest CER for the validation set. 
# 3. Also apply the prediction probability of the i-th validation observation, 
#    The threshold s starts from 0.1 to 0.9 increased by 0.01 
# 4. Find the optimal threshold from the validation set 
# ** If we have a tie of CER, select the smaller value of interaction.depth 
# ** Take an average of thresholds that have the same smallest CER. 
# 5. Apply the boosting model with the optimal number of interaction.depth and the optimal value of thresholds to the test 
# 6. Compute the averaged test CER 
# ** Fix the number of trees as 100 (n.trees=100 in the gbm function). 

thre <- seq(0.1, 0.9, 0.01)
d <- seq(1, 4, 1)

y <- as.numeric(y)-1 
tran_data <- data.frame(y = y, x.tran[,,i])
vald_data <- data.frame(y = y, x.vald[,,i])
test_data <- data.frame(y = y, x.test[,,i])

score1 <- matrix(NA, length(thre), length(d))
for (i in 1:length(d)) { 
  boost <- gbm(y ~., data=tran_data, n.trees=100, distribution="bernoulli", interaction.depth=d[i])
  for (j in 1:length(thre)) {
    pred_boost <- predict(boost, newdata=vald_data, n.trees=100)
    decision <- rep(0, 200) 
    decision[pred_boost > thre[j]] <- 1
    score1[j, i] <- mean(decision!=y)
  }
}

d_wh <- apply(score1, 1, which.min)
score2 <- matrix(0, length(thre), 1) 
for (i in 1:length(thre)) { 
  score2[i,] <- score1[i, d_wh[i]]  
}
score_wh <- which(score2 == min(score2)) 
score_thre <- mean(thre[score_wh])

boost_test <- gbm(y ~., data=tran_data, n.trees=100, distribution="bernoulli", interaction.depth=d_wh[which.min(score_wh)])
pred_boost_test <- predict(boost, newdata=test_data, n.trees=100)
decision_test <- rep(0, 200)
decision_test[pred_boost_test > score_thre] <- 1
score_test <- mean(decision_test!=y)
score_test
