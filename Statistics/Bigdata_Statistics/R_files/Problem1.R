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

fun <- function(type=c('lr', 'lda', 'qda', 'nb'), thre, x, y, v, w, pred, pred_test) { 
  score <- matrix(NA, length(thre)) 
  
  for (i in 1:length(thre)) { 
    decision = rep("-1", 200)
    decision[pred > thre[i]] = "1"
    score[i,] <- mean(decision != v)
  }
  
  score_wh <- which(score == min(score))
  score_thre <- mean(thre[score_wh])
  
  decision_test <- rep("-1", 200)
  decision_test[pred_test > score_thre] = "1"
  score_test <- mean(decision_test != w)
  
  return(score_test)
}

# ======================== Problem 01 ==========================
# Model to construct : 
# 1. LR(Logistic Regression) 
# 2. LDA(Linear Discriminant Analysis) 
# 3. QDA(Quadratic Discriminant Analysis) 
# 4. NB(Naive Bayes) 

# What need to apply 
# 1. Compute the prediction probability of the i-th validation observation
# 2. Classify p(y_i=1|x) > s, s begins from 0.1 to 0.9 increased by 0.01 
# 3. Find the optimal threshold from the validation set 
# ** If the multiple thresholds have the same smallest CER in the validation set, 
# take the sample mean of the multiple thresholds as the optimal threshold
# 4. Find the averaged CERs of the test sets of four classifiers

thre <- seq(0.1, 0.9, 0.01)

CER1 <- matrix(0, K, 4); colnames(CER1) <- c("LR", "LDA", "QDA", "NB")
for (i in 1:K) {
  tran_data <- data.frame(y = y, x.tran[,,i])
  vald_data <- data.frame(y = y, x.vald[,,i])
  test_data <- data.frame(y = y, x.test[,,i])
  
  LR <- glm(y ~., data=tran_data, family="binomial")
  pred_LR <- predict(LR, vald_data, type="response") 
  pred_LR_test <- predict(LR, test_data, type="response")
  
  LDA <- lda(y ~., data=tran_data)
  pred_LDA <- predict(LDA, vald_data)$posterior[,2]
  pred_LDA_test <- predict(LDA, test_data)$posterior[,2]
  
  QDA <- qda(y ~., data=tran_data)
  pred_QDA <- predict(QDA, vald_data)$posterior[,2]
  pred_QDA_test <- predict(QDA, test_data)$posterior[,2]
  
  NB <- naiveBayes(y ~., data=tran_data)
  pred_NB <- predict(NB, vald_data, type="raw")[,2]
  pred_NB_test <- predict(NB, test_data, type="raw")[,2]
  
  CER1[i, 1] <- fun(type="lr", thre=thre, v=y, pred=pred_LR, w=y, pred_test=pred_LR_test)
  CER1[i, 2] <- fun(type="lda", thre=thre, v=y, pred=pred_LDA, w=y, pred_test=pred_LDA_test)
  CER1[i, 3] <- fun(type="qda", thre=thre, v=y, pred=pred_QDA, w=y, pred_test=pred_QDA_test)
  CER1[i, 4] <- fun(type="nb", thre=thre, v=y, pred=pred_NB, w=y, pred_test=pred_NB_test)
}
CER1
