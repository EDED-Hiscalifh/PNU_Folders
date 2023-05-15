# 1. Linear Models 

## 1.1 OLS models 

set.seed(123)
n <- 100
# pp : pp means the number of samples 
pp <- c(10, 50, 80, 95, 97, 98, 99)
B <- matrix(0, 100, length(pp))


for (i in 1:100) {
  for (j in 1:length(pp)) {
    beta <- rep(0, pp[j])
    # beta1 == 1, beta0, beta2, ... betap -> 0 
    beta[1] <- 1
    x <- matrix(rnorm(n*pp[j]), n, pp[j])
    # x %*% beta is same as x[, 1]
    y <- x %*% beta + rnorm(n)
    g <- lm(y~x)
    B[i,j] <- g$coef[2]
  }
}
boxplot(B, col="orange", boxwex=0.6, ylab="Coefficient estimates",
        names=pp, xlab="The number of predictors", ylim=c(-5,5))
abline(h=1, col=2, lty=2, lwd=2)
apply(B, 2, mean)
apply(B, 2, var)

## 1.2 Best subset selection 

library(ISLR)
names(Hitters)
dim(Hitters)
sum(is.na(Hitters$Salary))
Hitters <- na.omit(Hitters)
dim(Hitters)
sum(is.na(Hitters))
library(leaps)
fit <- regsubsets(Salary ~ ., Hitters)
summary(fit)
sg <- summary(fit)
names(sg)
dim(sg$which)
sg$which
plot(fit)
plot(fit, scale="Cp")

# 2. Choosing the optimal models 

## 2.1 Find the best model considering same model size 

big <- regsubsets(Salary ~ ., data=Hitters, nvmax=19, nbest=10)
sg <- summary(big)
dim(sg$which)
sg.size <- as.numeric(rownames(sg$which))
table(sg.size)
sg.rss <- tapply(sg$rss, sg.size, min)
w1 <- which.min(sg.rss)
sg.rsq <- tapply(sg$rsq, sg.size, max)
w2 <- which.max(sg.rsq)
par(mfrow=c(1,2))
plot(1:19, sg.rss, type="b", xlab="Number of Predictors",
     ylab="Residual Sum of Squares", col=2, pch=19)
points(w1, sg.rss[w1], pch="x", col="blue", cex=2)
plot(1:19, sg.rsq, type="b", xlab="Number of Predictors",
     ylab=expression(R^2), col=2, pch=19)
points(w2, sg.rsq[w2], pch="x", col="blue", cex=2)

## 2.2 Find the best model considering different model size 

sg.cp <- tapply(sg$cp, sg.size, min)
w3 <- which.min(sg.cp)
sg.bic <- tapply(sg$bic, sg.size, min)
w4 <- which.min(sg.bic)
sg.adjr2 <- tapply(sg$adjr2, sg.size, max)
w5 <- which.max(sg.adjr2)
par(mfrow=c(1,3))
plot(1:19, sg.cp, type="b", xlab ="Number of Predictors",
     ylab=expression(C[p]), col=2, pch=19)
points(w3, sg.cp[w3], pch="x", col="blue", cex=2)
plot(1:19, sg.bic, type="b", xlab ="Number of Predictors",
     ylab="Bayesian information criterion", col=2, pch=19)
points(w4, sg.bic[w4], pch="x", col="blue", cex=2)
plot(1:19, sg.adjr2, type="b", xlab ="Number of Predictors",
     ylab=expression(paste("Adjusted ", R^2)), col=2, pch=19)
points(w5, sg.adjr2[w5], pch="x", col="blue", cex=2)

## 2.3 Find the best model considering validation set 

library(ISLR)
library(leaps)
names(Hitters)
Hitters <- na.omit(Hitters)

# Train-test split : Bootstrap (66%, 33%) 
set.seed(1234)
train <- sample(c(TRUE, FALSE), nrow(Hitters), replace=TRUE) 
test <- (!train) 

# Training model 
# Consider RSS and R^2 among models with same sample size : find M_k 
g1 <- regsubsets(Salary ~ ., data=Hitters[train, ], nvmax=19) 
test.mat <- model.matrix(Salary~., data=Hitters[test, ]) 
val.errors <- rep(NA, 19)
# Calculating validation error 
for (i in 1:19) {
  coefi <- coef(g1, id=i) 
  pred <- test.mat[, names(coefi)] %*% coefi
  val.errors[i] <- mean((Hitters$Salary[test]-pred)^2) 
}
val.errors
w <- which.min(val.errors) 

plot(1:19, val.errors, type="l", col="red",
     xlab="Number of Predictors", ylab="Validation Set Error")
points(1:19, val.errors, pch=19, col="blue")
points(w, val.errors[w], pch="x", col="blue", cex=2)

## 2.4 Find the best model considering K-fold CV 

library(ISLR)
names(Hitters)

## Define new "predict" function on regsubset
predict.regsubsets <- function(object, newdata, id, ...) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefi <- coef(object, id=id)
  xvars <- names(coefi)
  mat[, xvars] %*% coefi
}

# KFold Train-test split : Bootstrap (66%, 33%) using 10 folds 
set.seed(1234)
K <- 10 
folds <- sample(rep(1:K, length=nrow(Hitters))) 

# Initialize error matrix of every fold 
cv.errors <- matrix(NA, K, 19, dimnames=list(NULL, paste(1:19))) 

# Repeat calculation in ever 8 folds
for (k in 1:K) { 
  train <- sample(c(TRUE, FALSE), nrow(Hitters), replace=TRUE) 
  test <- (!train) 
  
  # Training model 
  # Consider RSS and R^2 among models with same sample size : find M_k 
  fit <- regsubsets(Salary ~ ., data=Hitters[folds!=k, ], nvmax=19) 
  
  # Calculating validation error 
  for (i in 1:19) {
    pred <- predict(fit, Hitters[folds==k, ], id=i) 
    cv.errors[k, i] <- mean((Hitters$Salary[folds==k]-pred)^2) 
  }
} 
apply(cv.errors, 2, mean)
K.ERR <- apply(cv.errors, 2, mean)
ww <- which.min(K.ERR)

# Visualize the test error
plot(1:19, K.ERR, type="l", col="red",
     xlab="Number of Predictors", ylab="Cross-Validation Error")
points(1:19, K.ERR, pch=19, col="blue")
points(ww, K.ERR[ww], pch="x", col="blue", cex=2)

## 2.5 Find the best model considering one-standard error rules 

# Train-test split 
set.seed(111)
n = nrow(Hitters)
folds <- sample(rep(1:K, length=n))
CVR.1se <- matrix(NA, n, 19)

# Train the model M_k on ith fold 
for (i in 1:K) {
  fit <- regsubsets(Salary~., Hitters[folds!=i, ], nvmax=19)
  # Calculate CVE of test sample 
  for (j in 1:19) {
    pred <- predict(fit, Hitters[folds==i, ], id=j)
    CVR.1se[folds==i, j] <- (Hitters$Salary[folds==i]-pred)^2
  }
}

# Calculate average based on One-Standard Error rule 
avg <- apply(CVR.1se, 2, mean)
se <- apply(CVR.1se, 2, sd)/sqrt(n)
PE <- cbind(avg - se, avg, avg + se)

# Visualize test error 
matplot(1:19, PE, type="b", col=c(1,2,1), lty=c(3,1,3), pch=20,
        xlab="Number of Predictors", ylab="Cross-Validation Error")
points(which.min(avg), PE[which.min(avg),2],
       pch="o",col="blue",cex=2)
up <- which(PE[,2] < PE[which.min(PE[,2]),3])
points(min(up), PE[min(up),2], pch="x", col="blue", cex=2) 

# 3. Variable selection methods 

## 3.1 Ridge regression 

library(glmnet)
# model.matrix convert original data into new matrix which form added with intercept.
x <- model.matrix(Salary~., Hitters)[, -1] 
y <- Hitters$Salary

# Grid Search for hyper parameter tuning lambda 
grid <- 10^seq(10, -2, length = 100) 
# Fit the ridge model : Default grid of lambda 
ridge.mod <- glmnet(x, y, alpha = 0, lambda=grid) 

# row for the number of parameters 
# col for the number of hyperparameter lambda 
dim(coef(ridge.mod)) 
ridge.mod$lambda 

# Calculate l_2 norm of each lambda 
# If lambda increase, l_2 norm decreaes 
# If lambda decreaes, l_2 norm increase 
ridge.mod$lambda[60]
coef(ridge.mod)[,60]
sqrt(sum(coef(ridge.mod)[-1, 60]^2))

# Calculate l2-norm based on lambda 
l2.norm <- apply(ridge.mod$beta, 2, function(t) sum(t^2))
x.axis <- cbind(log(ridge.mod$lambda), l2.norm)
colnames(x.axis) <- c("log.lambda", "L2.norm")
x.axis

# Visualize plot
par(mfrow=c(1,2))
plot(ridge.mod, "lambda", label=TRUE)
plot(ridge.mod, "norm", label=TRUE)

## 3.2 Lasso regression 

# lasso when the value of argument alpha is 1. 
lasso.mod <- glmnet(x, y, alpha=1) 
# the number of default value of lambda is 100. 
# However, considering complexity of parameter, our model us 80 lambdas.   
dim(coef(lasso.mod)) 

# Find the degree of freedom matrix 
las <- cbind(lasso.mod$lambda, lasso.mod$df) 
colnames(las) <- c("lambda", "df") 
las

# Find the beta
# The sum of beta which is not zero is same with the value of df. 
dim(lasso.mod$beta)
apply(lasso.mod$beta, 2, function(t) sum(t!=0))
apply(lasso.mod$beta, 2, function(t) sum(abs(t)))

# Calculate l1-norm based on lambda 
l1.norm <- apply(lasso.mod$beta, 2, function(t) sum(abs(t)))
x.axis <- cbind(log(lasso.mod$lambda), l1.norm)
colnames(x.axis) <- c("log.lambda", "L1.norm")
x.axis

# Visualize plot
par(mfrow=c(1,2))
plot(lasso.mod, "lambda", label=TRUE)
plot(lasso.mod, "norm", label=TRUE)

# 4. Selecting tuning parameters 

## 4.1 Validation set 

# Make dataset 
library(glmnet)
library(ISLR) 
names(Hitters) 
Hitters <- na.omit(Hitters) 

set.seed(123)
x <- model.matrix(Salary~., Hitters)[, -1] 
y <- Hitters$Salary

# Train-Test Split
train <- sample(1:nrow(x), nrow(x)/3) 
test <- (-train) 
y.test <- y[test]

# Hyperparameter tuning 
grid <- 10^seq(10, -2, length=100) 
r1 <- glmnet(x[train, ], y[train], alpha=0, lambda=grid)
ss <- 0:(length(r1$lambda)-1) 
Err <- NULL

# Cross validation Error for test sample 
for (i in 1:length(r1$lambda)) { 
  r1.pred <- predict(r1, s=ss[i], newx=x[test, ])
  Err[i] <- mean((r1.pred - y.test)^2) 
} 
wh <- which.min(Err) 
lam.opt <- r1$lambda[wh] 

# Get full model with optimized hyperparmeter 
r.full <- glmnet(x, y, alpha=0, lambda=grid) 
r.full$beta[, wh] 
predict(r.full, type="coefficients", s=lam.opt) 

## 4.2 K-fold Cross Validation 

set.seed(1234)
cv.r <- cv.glmnet(x, y, alpha=0, nfolds=10)
names(cv.r) 
# cvm : The mean value of cross validation -> CVE 
# cvsd : The standard deviation of cross validation -> One-standard error 
# cvup : The upperbound of CVE -> cvm + cvsd 
# cvlo : The lowerbound of CVE -> cvm - cvsd 
# lambda.min : The lambda which optimize input model 
# lambda.1se : The lambda which optimize imput model based on one-standard error 

cbind(cv.r$cvlo, cv.r$cvm, cv.r$cvup)
# Scatter plot based on One-Standard error 
# left vertix line : log(lambda.min) 
# right vertix line(more shrinked model) : log(lambda.1se) 
plot(cv.r) 

which(cv.r$lambda==cv.r$lambda.min)
which(cv.r$lambda==cv.r$lambda.1se)
# 100, 54 -> lambda.min < lambda.1se 

b.min <- predict(cv.r, type="coefficients", s=cv.r$lambda.min)
b.1se <- predict(cv.r, type="coefficients", s=cv.r$lambda.1se)

# calculate l1-norm
# calculate sum(b.min!=0) - 1 to get l2-norm 
cbind(b.min, b.1se)
c(sum(b.min[-1]^2), sum(b.1se[-1]^2))
# sum(b.min[-1]^2) > sum(b.1se[-1]^2) 