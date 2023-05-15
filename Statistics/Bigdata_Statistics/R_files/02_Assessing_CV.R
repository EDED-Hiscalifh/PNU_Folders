# 3. Assessing model accuracy 

set.seed(12345)
## Simulate x and y based on a known function
fun1 <- function(x) -(x-100)*(x-30)*(x+15)/13^4+6
x <- runif(50,0,100)
y <- fun1(x) + rnorm(50)

## Plot linear regression and splines
par(mfrow=c(1,2))
plot(x, y, xlab="X", ylab="Y", ylim=c(1,13))
plot(x, y, xlab="X", ylab="Y", ylim=c(1,13))
lines(sort(x), fun1(sort(x)), col=1, lwd=2)
abline(lm(y~x)$coef, col="orange", lwd=2)
lines(smooth.spline(x,y, df=5), col="blue", lwd=2)
lines(smooth.spline(x,y, df=23), col="green", lwd=2)
legend("topleft", lty=1, col=c(1, "orange", "blue", "green"),
       legend=c("True", "df = 1", "df = 5", "df =23"),lwd=2)


set.seed(45678)
## Simulate training and test data (x, y)
tran.x <- runif(50,0,100)
test.x <- runif(50,0,100)
tran.y <- fun1(tran.x) + rnorm(50)
test.y <- fun1(test.x) + rnorm(50)

## Compute MSE along with different df
df <- 2:40
MSE <- matrix(0, length(df), 2)
for (i in 1:length(df)) {
  tran.fit <- smooth.spline(tran.x, tran.y, df=df[i])
  MSE[i,1] <- mean((tran.y - predict(tran.fit, tran.x)$y)^2)
  MSE[i,2] <- mean((test.y - predict(tran.fit, test.x)$y)^2)
}

## Plot both test and training errors
matplot(df, MSE, type="l", col=c("gray", "red"),
        xlab="Flexibility", ylab="Mean Squared Error",
        lwd=2, lty=1, ylim=c(0,4))
abline(h=1, lty=2)
legend("top", lty=1, col=c("red", "gray"),lwd=2,
       legend=c("Test MSE", "Training MSE"))
abline(v=df[which.min(MSE[,1])], lty=3, col="gray")
abline(v=df[which.min(MSE[,2])], lty=3, col="red")

# 4. Cross Validation 

## 4.1 Validation-Set approach 

# Dataset Preparation 
library(ISLR) 
data(Auto) 
str(Auto) 
summary(Auto) 

# Extract target 
mpg <- Auto$mpg
horsepower <- Auto$horsepower

# set df 
dg <- 1:9
u <- order(horsepower) 

# Preview dataset 
par(mfrow=c(3,3))
for (k in 1:length(dg)) {
  g <- lm(mpg ~ poly(horsepower, dg[k]))
  plot(mpg~horsepower, col=2, pch=20, xlab="Horsepower",
       ylab="mpg", main=paste("dg =", dg[k]))
  lines(horsepower[u], g$fit[u], col="darkblue", lwd=3)
}

# Single Split 
set.seed(1)
n <- nrow(Auto)

## training set
tran <- sample(n, n/2)
MSE <- NULL
for (k in 1:length(dg)) {
  g <- lm(mpg ~ poly(horsepower, dg[k]), subset=tran)
  MSE[k] <- mean((mpg - predict(g, Auto))[-tran]^2)
}

dev.off()
# Visualization MSE_test
plot(dg, MSE, type="b", col=2, xlab="Degree of Polynomial",
     ylab="Mean Squared Error", ylim=c(15,30), lwd=2, pch=19)
abline(v=which.min(MSE), lty=2)