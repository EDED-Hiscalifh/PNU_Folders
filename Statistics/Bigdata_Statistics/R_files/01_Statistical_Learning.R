# 1. Statistical Learning 

## 1.1 [Ex] Advertising Data 

## Open the dataset linked to the book website
url.ad <- "https://www.statlearning.com/s/Advertising.csv"
Advertising <- read.csv(url.ad, h=T)
attach(Advertising)

## Least square fit for simple linear regression
par(mfrow = c(1,3))
plot(sales~TV, col=2, xlab="TV", ylab="Sales")
abline(lm(sales~TV)$coef, lwd=3, col="darkblue")

plot(sales~radio, col=2, xlab="Radio", ylab="Sales")
abline(lm(sales~radio)$coef, lwd=3, col="darkblue")

plot(sales~newspaper, col=2, xlab="Newspaper", ylab="Sales")
abline(lm(sales~newspaper)$coef, lwd = 3, col="darkblue")

# 2. Supervised Learning 

## 2.1 [Ex] Modeling 

## Indexing without index 
AD <- Advertising[, -1] 

## Multiple linear regression 
lm.fit <- lm(sales ~., AD) 
summary(lm.fit)
names(lm.fit) 
coef(lm.fit)
confint(lm.fit) 

## Visualizaing models 
par(mfrow=c(2,2))
plot(lm.fit) 

dev.off()
plot(predict(lm.fit), residuals(lm.fit))    # Residual vs Fitted  
plot(predict(lm.fit), rstudent(lm.fit))    
plot(hatvalues(lm.fit))
which.max(hatvalues(lm.fit)) 

## 2.2 [Ex] Income data prediction 

## Load Datasets
url.in <- "https://www.statlearning.com/s/Income1.csv"
Income <- read.csv(url.in, h=T)

## Polynomial regression fit 
par(mfrow = c(1,2)) 
plot(Income~Education, col=2, pch=19, xlab="Years of Education", 
     ylab="Income", data=Income) 

g <- lm(Income ~ poly(Education, 3), data=Income) 
plot(Income~Education, col=2, pch=19, xlab="Years of Education", 
     ylab="Income", data=Income)
lines(Income$Education, g$fit, col="darkblue", lwd=4, ylab="Income", 
      xlab="Years of Education")

## Compare residuals
y <- Income$Income
mean((predict(g) - y)^2) 
mean(residuals(g)^2)

# Train error by different polynomials 
dist <- NULL
par(mfrow=c(3,4)) 
for (k in 1:12) { 
  g <- lm(Income ~ poly(Education, k), data=Income) 
  dist[k] <- mean(residuals(g)^2)
  plot(Income~Education, col=2, pch=19, xlab="Years of Education", ylab="Income",
       data=Income, main=paste("k =", k)) 
  lines(Income$Education, g$fit, col="darkblue", lwd=3, ylabe="Income", xlab="Years of Education")
}

# Mean squared distiance of different polynomials 
x11()
plot(dist, type="b", xlab="Degree of Polynomial", 
     ylab="Mean squared distance")