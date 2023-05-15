# =========== 1. Non-linear SVMs with radial kernel ==============

c
library(e1071)
set.seed(1)
x <- matrix(rnorm(200*2), ncol=2)
x[1:100, ] <- x[1:100, ] + 2
x[101:150, ] <- x[101:150, ] - 2
y <- c(rep(-1, 150), rep(1, 50))
dat <- data.frame(x, y=as.factor(y))
plot(x, col=y+3, pch=19)

# Training svm model with radial kernel with r=0.5, C=0.1
fit <- svm(y~.,data=dat, kernel="radial", gamma=0.5, cost=0.1)
plot(fit, dat)
summary(fit)

# Training svm model with radial kernel with r=0.5, C=5
fit <- svm(y~.,data=dat, kernel="radial", gamma=0.5, cost=5)
plot(fit, dat)
summary(fit)

# Visualize of test grid for radial kernel with r=0.5, C=1
fit <- svm(y~.,data=dat, kernel="radial", gamma=0.5, cost=1)
px1 <- seq(round(min(x[,1]),1), round(max(x[,1]),1), 0.1)
px2 <- seq(round(min(x[,2]),1), round(max(x[,2]),1), 0.1)
xgrid <- expand.grid(X1=px1, X2=px2)
ygrid <- as.numeric(predict(fit, xgrid))
ygrid[ygrid==1] <- -1
ygrid[ygrid==2] <- 1
plot(xgrid, col=ygrid+3, pch = 20, cex = .2)
points(x, col = y+3, pch = 19)
pred <- predict(fit, xgrid, decision.values=TRUE)
func <- attributes(pred)$decision
contour(px1, px2, matrix(func, length(px1), length(px2)),
level=0, col="purple", lwd=2, lty=2, add=TRUE)


# =============== 2. Optimizing non-linear SVMs(radial kernel) using validation set ============== 
# Calculate missclassification error of validation set 
# Separate training and test sets 
set.seed(1234)
tran <- sample(200, 100)
test <- setdiff(1:200, tran)

# Training with hyperparameter tuning of gamma, C
gamma <- c(0.5, 1, 5, 10)
cost <- c(0.01, 1, 10, 100)
R <- NULL
for (i in 1:length(gamma)) {
  for (j in 1:length(cost)) {
    svmfit <- svm(y~., data=dat[tran, ], kernel="radial",
                  gamma=gamma[i] , cost=cost[j])
    pred <- predict(svmfit, dat[test, ])
    R0 <- c(gamma[i], cost[j], mean(pred!=dat[test, "y"]))
    R <- rbind(R, R0)
  }
}

# Check results 
colnames(R) <- c("gamma", "cost", "error")
rownames(R) <- seq(dim(R)[1])
R

# Training with hyperparameter tuning of gamma, C using tune function 
set.seed(1)
tune.out <- tune(svm, y~., data=dat[tran, ], kernel="radial",
                 ranges=list(gamma=gamma, cost=cost))
summary(tune.out)
tune.out$best.parameters

# Calculate missclassification error rate of test sets
pred <- predict(tune.out$best.model, dat[test,])
table(pred=pred, true=dat[test, "y"])
mean(pred!=dat[test, "y"])

# =============== 3. Optimizing non-linear SVMs(polynomial kernel) using validation set ============== 
degree <- c(1, 2, 3, 4)
R <- NULL

for (i in 1:length(degree)) {
  for (j in 1:length(cost)) {
    svmfit <- svm(y~., data=dat[tran, ], kernel="polynomial",
                  degree=degree[i] , cost=cost[j])
    pred <- predict(svmfit, dat[test, ])
    R0 <- c(degree[i], cost[j], mean(pred!=dat[test, "y"]))
    R <- rbind(R, R0)
  }
}
colnames(R) <- c("degree", "cost", "error")
rownames(R) <- seq(dim(R)[1])
R


tune.out <- tune(svm, y~., data=dat[tran, ], kernel="polynomial",
                 ranges=list(degree=degree, cost=cost))
summary(tune.out)
tune.out$best.parameters

pred <- predict(tune.out$best.model, dat[test,])
table(pred=pred, true=dat[test, "y"])
mean(pred!=dat[test, "y"])

# =============== 4. Optimizing non-linear SVMs(sigmoid kernel) using validation set ============== 
R <- NULL
for (i in 1:length(gamma)) {
  for (j in 1:length(cost)) {
    svmfit <- svm(y~., data=dat[tran, ], kernel="sigmoid",
                  gamma=gamma[i] , cost=cost[j])
    pred <- predict(svmfit, dat[test, ])
    R0 <- c(gamma[i], cost[j], mean(pred!=dat[test, "y"]))
    R <- rbind(R, R0)
  }
}

colnames(R) <- c("gamma", "cost", "error")
rownames(R) <- seq(dim(R)[1])
R

tune.out <- tune(svm, y~., data=dat[tran, ], kernel="sigmoid",
                        ranges=list(gamma=gamma, cost=cost))
summary(tune.out)
tune.out$best.parameters

pred <- predict(tune.out$best.model, dat[test,])
table(pred=pred, true=dat[test, "y"])
mean(pred!=dat[test, "y"])

# ================= 5. Simulation Study using different kernels of 20 replications =============
# Set reps and RES matrix 
set.seed(123)
N <- 20
RES <- matrix(0, N, 3)
colnames(RES) <- c("radial", "poly", "sigmoid")

# Training model with calculate missclassification error rate 
for (i in 1:N) {
  tran <- sample(200, 100)
  test <- setdiff(1:200, tran)
  tune1 <- tune(svm, y~., data=dat[tran, ], kernel="radial",
                ranges=list(gamma=gamma, cost=cost))
  pred1 <- predict(tune1$best.model, dat[test,])
  RES[i, 1] <- mean(pred1!=dat[test, "y"])
  tune2 <- tune(svm, y~., data=dat[tran, ], kernel="polynomial",
                ranges=list(degree=degree, cost=cost))
  pred2 <- predict(tune2$best.model, dat[test,])
  RES[i, 2] <- mean(pred2!=dat[test, "y"])
  tune3 <- tune(svm, y~., data=dat[tran, ], kernel="sigmoid",
                ranges=list(gamma=gamma, cost=cost))
  pred3 <- predict(tune3$best.model, dat[test,])
  RES[i, 3] <- mean(pred3!=dat[test, "y"])
}
# Check statistical reports 
apply(RES, 2, summary)