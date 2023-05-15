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

# ======================== Problem 02 ==========================
# What need to apply 
# 1. Fit a classification tree for the training set 
# 2. Find the optimal size to prune the tree, using prune.misclass function for the validation set
# 3. Apply the optimal tree to the test set and compute the average test CER.
# ** If the optimal tree has only one node, fix CER=0.5 

tran_data <- data.frame(y = y, x.tran[,,1])
vald_data <- data.frame(y = y, x.vald[,,1])
test_data <- data.frame(y = y, x.test[,,1])

d <- seq(1, 25, 1)

score <- matrix(NA, length(d)) 
tree.tran <- tree(y~., data=tran_data)
for (i in 2:length(d)) {
  prune.tree.tran <- prune.misclass(tree.tran, best=d[i])
  decision <- predict(prune.tree.tran, vald_data, type="class")
  score[i,] <- mean(decision!=y)
}

score_wh <- which.min(score)

if (score_wh == 1) {
  score_test <- 0.5
} else {
  prune.tree_test <- prune.misclass(trees, best=score_wh)
  decision_test <- predict(prune.tree_test, test_data, type="class")
  score_test <- mean(decision_test!=y)
}
score_test
