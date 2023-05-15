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

# Importing library
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

thre <- seq(0.1, 0.9, 0.01)
d <- seq(1, 4, 1)

y <- ifelse(y=="1", 1, -1)
tran_data <- data.frame(y = y, x.tran[,,1])
vald_data <- data.frame(y = y, x.vald[,,1])
test_data <- data.frame(y = y, x.test[,,1])

score1 <- matrix(NA, length(thre), length(d))
for (i in 1:length(d)) { 
  ada <- adaboost(X=as.matrix(tran_data[, -1]), y=y, tree_depth=d[i])
  for (j in 1:length(thre)) {
    pred_boost <- predict(ada, as.matrix(vald_data[, -1]))
    decision <- rep(-1, 200) 
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

ada_test <- adaboost(X=as.matrix(tran_data[, -1]), y=y, tree_depth=d_wh[which.min(score_wh)])
pred_ada_test <- predict(ada_test, as.matrix(test_data[, -1]))
decision_test <- rep(-1, 200)
decision_test[pred_ada_test > score_thre] <- 1
score_test <- mean(decision_test!=y)
score_test