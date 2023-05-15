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

# Function for solving problem 
fun <- function(type=c('lr', 'lda', 'qda', 'nb'), thre, m, d, x, y, v, w, pred, pred_test, tran, vald, test) { 
  # Problem1 
  if (type == 'lr' | type == 'lda' | type == 'qda' | type == 'nb') {
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
  } else if(type == 'tree') {
    score <- matrix(NA, length(d)) 
    trees <- tree(y~., data=tran)
    for (i in 1:length(d)) {
      prune.trees <- prune.misclass(trees, best=d[i])
      decision <- predict(prune.trees, vald, type="class")
      score[i,] <- mean(decision!=v)
    }
    
    score_wh <- which.min(score)
    
    if (score_wh == 1) {
      return(0.5)
    } else {
      prune.tree_test <- prune.misclass(trees, best=score_wh)
      decision_test <- predict(prune.tree_test, test, type="class")
      score_test <- mean(decision_test!=w)
      return(score_test)
    }
  } else if (type == 'rf') {
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
    
    RF_test <- randomForest(x=tran[, -1], y=tran[, 1], 
                       xtest=test[, -1], ytest=test[, 1],
                       ntree=500, mtry=score_m, importance=TRUE)
    RF_test.conf <- RF_test$test$confusion[1:2,1:2]
    score_test <- 1 - sum(diag(RF_test.conf))/sum(RF_test.conf)
    return(score_test)
  } else if (type=='boosting') { 
    score1 <- matrix(NA, length(thre), length(d))
    for (i in 1:length(d)) { 
      boost <- gbm(y ~., data=tran, n.trees=100, distribution="bernoulli", interaction.depth=d[i])
      for (j in 1:length(thre)) {
        pred_boost <- predict(boost, newdata=vald, n.trees=100)
        decision <- rep(0, 200) 
        decision[pred_boost > thre[j]] <- 1
        score1[j, i] <- mean(decision!=v)
      }
    }
    
    d_wh <- apply(score1, 1, which.min)
    score2 <- matrix(0, length(thre), 1) 
    for (i in 1:length(thre)) { 
      score2[i,] <- score1[i, d_wh[i]]  
    }
    score_wh <- which(score2 == min(score2)) 
    score_thre <- mean(thre[score_wh])
    
    boost_test <- gbm(y ~., data=tran, n.trees=100, distribution="bernoulli", interaction.depth=d_wh[which.min(score_wh)])
    pred_boost_test <- predict(boost_test, newdata=test, n.trees=100)
    decision_test <- rep(0, 200)
    decision_test[pred_boost_test > score_thre] <- 1
    score_test <- mean(decision_test!=w)
    return(score_test) 
  } else if (type == 'adaboost') {
    score1 <- matrix(NA, length(thre), length(d))
    for (i in 1:length(d)) { 
      ada <- adaboost(X=as.matrix(tran[, -1]), y=y, tree_depth=d[i])
      for (j in 1:length(thre)) {
        pred_boost <- predict(ada, as.matrix(vald[, -1]))
        decision <- rep(-1, 200) 
        decision[pred_boost > thre[j]] <- 1
        score1[j, i] <- mean(decision!=v)
      }
    }
    
    d_wh <- apply(score1, 1, which.min)
    score2 <- matrix(0, length(thre), 1) 
    for (i in 1:length(thre)) { 
      score2[i,] <- score1[i, d_wh[i]]  
    }
    score_wh <- which(score2 == min(score2)) 
    score_thre <- mean(thre[score_wh])
    
    ada_test <- adaboost(X=as.matrix(tran[, -1]), y=y, tree_depth=d_wh[which.min(score_wh)])
    pred_ada_test <- predict(ada_test, as.matrix(test[, -1]))
    decision_test <- rep(-1, 200)
    decision_test[pred_ada_test > score_thre] <- 1
    score_test <- mean(decision_test!=w)
    score_test
  }
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

# ======================== Problem 02 ==========================
# What need to apply 
# 1. Fit a classification tree for the training set 
# 2. Find the optimal size to prune the tree, using prune.misclass function for the validation set
# 3. Apply the optimal tree to the test set and compute the average test CER.
# ** If the optimal tree has only one node, fix CER=0.5 

d <- seq(2, 10, 1)
CER2 <- matrix(0, K, 1); colnames(CER2) <- c("tree") 
for (i in 1:K) {
  tran_data <- data.frame(y = y, x.tran[,,i])
  vald_data <- data.frame(y = y, x.vald[,,i])
  test_data <- data.frame(y = y, x.test[,,i])
  CER2[i,] <- fun(type='tree', d=d, v=y, w=y, tran=tran_data, vald=vald_data, test=test_data)
}
CER2

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

d <- seq(1, 4, 1)
CER4 <- matrix(0, K, 1); colnames(CER4) <- c("Boosting")
y <- as.numeric(y)-1
for (i in 1:K) { 
  tran_data <- data.frame(y = y, x.tran[,,i])
  vald_data <- data.frame(y = y, x.vald[,,i])
  test_data <- data.frame(y = y, x.test[,,i])
  CER4[i,] <- fun(type='boosting', thre=thre, d=d, v=y, w=y, tran=tran_data, vald=vald_data, test=test_data)
}
CER4

# ======================== Problem 05 ==========================
# What need to apply 
# 1. Repeat Q4, replacing gbm function by adaboost function in an R package JOUSBoost 
# ** Need to find the optimal tree_depth among 1 - 4 and the optimal value of the threshold from the valdiation set
# ** Do not change other arguments in the function, including n_rounds 

CER5 <- matrix(0, K, 1); colnames(CER5) <- c("Adaboost")
y <- ifelse(y=="1", 1, -1)
for (i in 1:K) { 
  tran_data <- data.frame(y = y, x.tran[,,i])
  vald_data <- data.frame(y = y, x.vald[,,i])
  test_data <- data.frame(y = y, x.test[,,i])
  # CER5[i,] <- fun(type='adaboost', thre=thre, d=d, y=y, v=y, w=y, tran=tran_data, vald=vald_data, test=test_data)
}
CER5

# ======================== Problem 06 ==========================
# What need to apply 
# 1. Apply a glmnet function to the training set using family = "binomial" 
# 2. For each lambda value, compute the prediction probability of the i - th validation observation
# ** The thresholds s starts from 0.1 to 0.9 increased by 0.01 
# 3. Find the optimal lambda and threshold that minimize the CER of the validation set. 
# ** If we have a tie of CER, pick up a smaller lambda and then take an average of the thresholds that has the smallest CER
# 4. Apply the lasso with the optimal lambda and the optimal value of threshold to the test set 
# 5. Compute the averaged test CER 

CER6 <- matrix(0, K, 1); colnames(CER6) <- c('Lasso') 
lambda <- seq(0.0007, 0.26, length.out=200)
for (i in 1:K) {
  tran_data <- data.frame(y = y, x.tran[,,i])
  vald_data <- data.frame(y = y, x.vald[,,i])
  test_data <- data.frame(y = y, x.test[,,i])
  CER6[i,] <- fun(type='lasso', thre=thre, lamb=lambda, y=y, v=y, w=y, tran=tran_data, vald=vald_data, test=test_data)
}
CER6

# ======================== Problem 07 ==========================
# What need to apply 
# 1. Draw a side-by-side boxplot where 9 methods are on the x-axis, CERs are on the y-axis 
# 2. Find which method has the smallest mean/variance of the test CER?
RES <- cbind(CER1, CER2, CER3, CER4, CER5, CER6); colnames(RES) <- cbind("LR", "LDA", "QDA", "NB", "tree", "RF", "gbm", "adaboost", "lasso")
boxplot(RES)


apply(RES, 2, mean)
apply(RES, 2, var)
