# 1. Quantitative outcome 

# install.packages('glmnet')
library(glmnet)

# Generate X matrix and Y vector
sim.fun <- function(n, p, beta, family=c("gaussian", "binomial")) {
  family <- match.arg(family)
  if (family=="gaussian") {
    x <- matrix(rnorm(n*p), n, p)
    y <- x %*% beta + rnorm(n)
  }
  else {
    x <- matrix(rnorm(n*p), n, p)
    xb <- x %*% beta
    z <- exp(xb) / (1+exp(xb))
    u <- runif(n)
    y <- rep(0, n)
    y[z > u] <- 1
  }
  list(x=x, y=y)
}

# Configure dimension of n, p and beta
set.seed(1234)
n <- 200
p <- 2000
beta <- rep(0, p)
beta[1:20] <- runif(20, -1, 1)

# Generate x and y 
sim <- sim.fun(n, p, beta)
x <- sim$x; y <- sim$y

# Fit the lasso with two different lambda values
# b hat is inferred regression coefficients : M1, M2 
g <- cv.glmnet(x, y, alpha=1, nfolds = 10)
bhat1 <- coef(g, s="lambda.min")
bhat2 <- coef(g, s="lambda.1se")
# Check coefficients with value is not 0. 
wh1 <- which(as.matrix(bhat1)!=0)
w1 <- wh1[-1]-1
wh2 <- which(as.matrix(bhat2)!=0)
w2 <- wh2[-1]-1

# Compute ordinary least square estimates (unbiased estimates)
bhat3 <- bhat4 <- bhat5 <- rep(0, p+1)
bhat3[wh1] <- lm(y ~ x[, w1])$coef
bhat4[wh2] <- lm(y ~ x[, w2])$coef
bhat5[1:21] <- lm(y ~ x[, 1:20])$coef

set.seed(56789)
# Generate test sets
test <- sim.fun(n, p, beta)
xt <- cbind(1, test$x)
yt <- test$y

# Test set prediction errors of 6 coefficient estimates
mean((yt - xt %*% bhat1)^2) # lasso_lambda.min (M1)
mean((yt - xt %*% bhat2)^2) # lasso_lambda.1se (M2)
mean((yt - xt %*% bhat3)^2) # least square_lambda.min (M3)
mean((yt - xt %*% bhat4)^2) # least square_lambda.1se (M4)
mean((yt - xt %*% bhat5)^2) # least square_nonzero beta (M5)
mean((yt - xt %*% c(0, beta))^2) # true beta (M6)

# Calculate TE repeatedly
set.seed(1)
# Generate new test sets 100 times
K <- 100
pred <- matrix(NA, K, 6)
for (i in 1:K) {
  test <- sim.fun(n, p, beta)
  xt <- cbind(1, test$x)
  yt <- test$y
  
  pred[i, 1] <- mean((yt - xt %*% bhat1)^2)
  pred[i, 2] <- mean((yt - xt %*% bhat2)^2)
  pred[i, 3] <- mean((yt - xt %*% bhat3)^2)
  pred[i, 4] <- mean((yt - xt %*% bhat4)^2)
  pred[i, 5] <- mean((yt - xt %*% bhat5)^2)
  pred[i, 6] <- mean((yt - xt %*% c(0, beta))^2)
}

apply(pred, 2, mean)
boxplot(pred, col=c(2,2,4,4,3,"orange"), boxwex=0.6,
        names=c("M1", "M2", "M3", "M4", "M5", "M6"),
        ylab="Prediction Error")  

# 2. Binary Outcome 

set.seed(111)
n <- 200
p <- 2000
beta <- rep(0, p)
beta[1:20] <- runif(20, -1, 1)
sim <- sim.fun(n, p, beta, family="binomial")
x <- sim$x; y <- sim$y

## Classification Error
class.fun <- function(test.x, test.y, beta, k=0.5) {
  # xb : calculate x % coefficients
  xb <- test.x %*% beta
  # exb : inferred probability of xb 
  exb <- exp(xb) / (1 + exp(xb))
  y <- rep(0, length(test.y))
  y[as.logical(exb > k)] <- 1
  min(mean(test.y!=y), mean(test.y!=(1-y)))
}

g <- cv.glmnet(x, y, alpha=1, nfolds = 10, family="binomial")
bhat1 <- coef(g, s="lambda.min")
bhat2 <- coef(g, s="lambda.1se")
wh1 <- which(as.matrix(bhat1)!=0)
wh2 <- which(as.matrix(bhat2)!=0)
bhat3 <- bhat4 <- bhat5 <- rep(0, p+1)
w1 <- wh1[-1]-1; w2 <- wh2[-1]-1
bhat3[wh1] <- glm(y ~ x[, w1], family="binomial")$coef
bhat4[wh2] <- glm(y ~ x[, w2], family="binomial")$coef
bhat5[1:21] <- glm(y ~ x[, 1:20], family="binomial")$coef

# Generate test sets
set.seed(56789)
test <- sim.fun(n, p, beta, family="binomial")
xt <- cbind(1, test$x); yt <- test$y

# Prediction error comparison
class.fun(xt, yt, bhat1) # lasso_lambda.min (M1)
class.fun(xt, yt, bhat2) # lasso_lambda.1se (M2)
class.fun(xt, yt, bhat3) # least square_lambda.min (M3)
class.fun(xt, yt, bhat4) # least square_lambda.1se (M4)
class.fun(xt, yt, bhat5) # least square_nonzero beta (M5)
class.fun(xt, yt, c(0, beta)) # true beta (M6)

# Calculate TE repeatedly
set.seed(35791)

# Generate new test sets 100 times
K <- 100
pred <- matrix(NA, K, 6)
for (i in 1:K) {
  test <- sim.fun(n, p, beta, family="binomial")
  xt <- cbind(1, test$x)
  yt <- test$y
  
  pred[i, 1] <- class.fun(xt, yt, bhat1)
  pred[i, 2] <- class.fun(xt, yt, bhat2)
  pred[i, 3] <- class.fun(xt, yt, bhat3)
  pred[i, 4] <- class.fun(xt, yt, bhat4)
  pred[i, 5] <- class.fun(xt, yt, bhat5)
  pred[i, 6] <- class.fun(xt, yt, c(0, beta))
}

apply(pred, 2, mean)
boxplot(pred, col=c(2,2,4,4,3,"orange"), boxwex=0.6,
        names=c("M1", "M2", "M3", "M4", "M5", "M6"),
        ylab="Prediction Error")  