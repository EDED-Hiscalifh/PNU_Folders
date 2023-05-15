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
lambda <- seq(0.00007, 0.26, length.out=200)

lasso <- glmnet(x.tran[,,1], y, alpha=1, lambda=lambda, family="binomial")

score1 <- matrix(NA, length(thre), length(lambda))
for (i in 1:length(lambda)) { 
  prob <- predict(lasso, x.vald[,,1], s=lasso$lambda[i], type="response")
  for (j in 1:length(thre)) {
    decision <- rep("-1", 200) 
    decision[prob > thre[j]] <- "1"
    score1[j, i] <- mean(decision!=y)
  }
}

l_wh <- apply(score1, 1, which.min)
score2 <- matrix(0, length(thre), 1) 
for (i in 1:length(thre)) { 
  score2[i,] <- score1[i, l_wh[i]]  
}
score_wh <- which(score2 == min(score2)) 
score_thre <- mean(thre[score_wh])

prob_lasso_test <- predict(lasso, s=lasso$lambda[l_wh[which.min(score_wh)]], x.test[,,1])
decision_test <- rep("-1", 200)
decision_test[prob_lasso_test > score_thre] <- "1"
score_test <- mean(decision_test!=y)
score_test