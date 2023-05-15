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

fun <- function(type=c('lr', 'lda', 'qda', 'nb', 'rf'), thre, m, x, y, v, w, pred, pred_test, tran, vald, test) { 
  score <- matrix(NA, length(m)) 
  
  for (i in 1:length(m)) { 
    RF <- randomForest(x=tran[, -1], y=tran[, 1], 
                       xtest=vald[, -1], ytest=vald[, 1], 
                       ntree=500, mtry=4, importance=TRUE)
    RF.conf <- RF$test$confusion[1:2,1:2]
    score[i,] <- 1 - sum(diag(RF.conf))/sum(RF.conf)
  }
  
  score_wh <- which.min(score)
  score_m <- mean(m[score_wh])
  
  RF <- randomForest(x=tran[, -1], y=tran[, 1], 
                     xtest=test[, -1], ytest=test[, 1],
                     ntree=500, mtry=score_m, importance=TRUE)
  RF.conf <- RF$test$confusion[1:2,1:2]
  score_test <- 1 - sum(diag(RF.conf))/sum(RF.conf)
  return(score_test)
}

# ======================== Problem 03 ==========================
# What need to apply 
# 1. Fit a random forest for the training set where the number of predictors is 1 ... 10.
# 2. Find the optimal number of predictors which has the smallest CER for the validation set 
# ** If we have a tie of CER, just select a smaller value of the number of predictors.
# 3. Apply random forest with the optimal number of predictors to the test set 
# 4. Compute the averaged test CER. 
# ** Fix the number of trees as 500 (n.tree=500 in the randomForest function).

m <- seq(1, 10, 1)

CER3 <- matrix(0, K, 1); colnames(CER3) <- c("RF")
for (i in 1:K) { 
  tran_data <- data.frame(y = y, x.tran[,,i])
  vald_data <- data.frame(y = y, x.vald[,,i])
  test_data <- data.frame(y = y, x.test[,,i])
  CER3[i,] <- fun(type='rf', m=m, tran=tran_data, vald=vald_data, test=test_data)
}
CER3