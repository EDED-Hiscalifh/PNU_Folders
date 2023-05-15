# Simple example (simulate data set)
set.seed(1)
x <- matrix(rnorm(20*2), ncol=2)
y <- c(rep(-1, 10), rep(1, 10))
x[y==1, ] <- x[y==1, ] + 1
plot(x, col=(3-y), pch=19, xlab="X1", ylab="X2")

# Support vector classifier with cost=10
library(e1071)
# y must be in format (-1, 1) 
dat <- data.frame(x, y=as.factor(y))
# Tuning parameter cost (inverse of C) 
svmfit <- svm(y~., data=dat, kernel="linear", cost=10,
              scale=FALSE)
plot(svmfit, dat)
summary(svmfit)

# SVM of tuning parameter (cost=0.1) 
svmfit <- svm(y~., data=dat, kernel="linear", cost=0.1,
              scale=FALSE)
svmfit$index
beta <- drop(t(svmfit$coefs)%*%x[svmfit$index,])
beta0 <- svmfit$rho

# Visualize results
plot(x, col=(3-y), pch=19, xlab="X1", ylab="X2")
points(x[svmfit$index, ], pch=5, cex=2)
abline(beta0 / beta[2], -beta[1] / beta[2])
abline((beta0 - 1) / beta[2], -beta[1] / beta[2], lty = 2)
abline((beta0 + 1) / beta[2], -beta[1] / beta[2], lty = 2)

# A function to create a grid of values or a lattice of values
make.grid <- function(x, n = 75) {
  ran <- apply(x, 2, range)
  x1 <- seq(from = ran[1,1], to = ran[2,1], length = n)
  x2 <- seq(from = ran[1,2], to = ran[2,2], length = n)
  expand.grid(X1 = x1, X2 = x2)
}
xgrid <- make.grid(x)
xgrid[1:76,]
## Classification of potential test set
ygrid <- predict(svmfit, xgrid)
plot(xgrid, col=c("red","blue")[as.numeric(ygrid)], pch=20,
     cex=.2)
points(x, col=y+3, pch=19)
points(x[svmfit$index, ], pch=5, cex=2)

# Cross validation to find the optimal tuning parameter (cost)
set.seed(1)
tune.out <- tune(svm, y~., data=dat, kernel="linear",
                 ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
summary(tune.out)
bestmod <- tune.out$best.model
summary(bestmod)

# Generate test set
set.seed(4321)
xtest <- matrix(rnorm(20*2), ncol=2)
ytest <- sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1, ] <- xtest[ytest==1, ] + 1
testdat <- data.frame(xtest, y=as.factor(ytest))

# Compute misclassification rate for the optimal model
ypred <- predict(bestmod, testdat)
table(predict=ypred, truth=testdat$y)
mean(ypred!=testdat$y)

# Prepare package 
library(mnormt)
library(e1071)

# Function for calculating Misclassification Rate of SVC
SVC.MCR <- function(x.tran, x.test, y.tran, y.test,
                    cost=c(0.01,0.1,1,10,100)) {
  dat <- data.frame(x.tran, y=as.factor(y.tran))
  testdat <- data.frame(x.test, y=as.factor(y.test))
  MCR <- rep(0, length(cost)+1)
  for (i in 1:length(cost)) {
    svmfit <- svm(y~., data=dat, kernel="linear",
                  cost=cost[i])
    MCR[i] <- mean(predict(svmfit, testdat)!=testdat$y)
  }
  tune.out <- tune(svm, y~., data=dat, kernel="linear",
                   ranges=list(cost=cost))
  pred <- predict(tune.out$best.model, testdat)
  MCR[length(cost)+1] <- mean(pred!=testdat$y)
  MCR
}

# Simulation test for 100 replications
set.seed(123)
K <- 100
RES <- matrix(NA, K, 6)
colnames(RES) <- c("0.01", "0.1" ,"1" , "10" ,"100", "CV")
for (i in 1:K) {
  x.A <- rmnorm(100, rep(0, 2), matrix(c(1,-0.5,-0.5,1),2))
  x.B <- rmnorm(100, rep(1, 2), matrix(c(1,-0.5,-0.5,1),2))
  x.tran <- rbind(x.A[1:50, ], x.B[1:50, ])
  x.test <- rbind(x.A[-c(1:50), ], x.B[-c(1:50), ])
  y.tran <- factor(rep(0:1, each=50))
  y.test <- factor(rep(0:1, each=50))
  RES[i,] <- SVC.MCR(x.tran, x.test, y.tran, y.test)
}
apply(RES, 2, summary)
boxplot(RES, boxwex=0.5, col=2:7,
        names=c("0.01", "0.1", "1", "10", "100", "CV"),
        main="", ylab="Classification Error Rates")