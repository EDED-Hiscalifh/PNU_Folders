# ======================================== [Ex] Credit Card Default Data ==================================== # 

# Import Dataset 
library(ISLR)
data(Default)
summary(Default)
attach(Default)

# Visualization of dataset 
plot(income ~ balance, xlab="Balance", ylab="Income",
     pch=c(1,3)[unclass(default)],
     col=c("lightblue","red")[unclass(default)])

# Check the dataset with same sample size 
set.seed(1234)
ss <- sample(which(default=="No"), sum(default=="Yes"))
ss <- c(ss, which(default=="Yes"))
us <- unclass(default[ss])
plot(income[ss] ~ balance[ss], xlab="Balance", pch=c(1,3)[us],
     col=c("lightblue","red")[us], ylab="Income")

# Boxplot of Default by Balance and Income 
par(mfrow=c(1,2))
boxplot(balance~default, col=c("lightblue","red"), boxwex=0.5, xlab="Default", ylab="Balance")
boxplot(income~default, col=c("lightblue","red"), boxwex=0.5, xlab="Default", ylab="Income")

# [Ex] Basic Model with glm function 
# Training model : balance 
g2 <- glm(default ~ balance, family="binomial") 
summary(g2)$coef

# Fitted values : The calculated probability 
g2$fit 

# Inverse logistic function : Calculating probability from new value 
ilogit <- function(x, coef) {
  exp(cbind(1, x) %*% coef) / (1 + exp(cbind(1, x) %*% coef)) 
} 
# Compared two values : This results will be same 
cbind(g2$fit, ilogit(balance, g2$coef))

# Calculate probability from new value x 
ilogit(1000, g2$coef) 

# Training model : student 
g3 <- glm(default ~ student, family="binomial") 
summary(g3$coef) 

# Probability when student "yes" 
ilogit(1, g3$coef)
# Probability when student "no" 
ilogit(0, g3$coef)

# ======================================== [Ex] Find optimized model based on K-fold CV ==================================== # 

# Train-Test split 
set.seed(1111)
n <- nrow(Default) 
train <- sample(1:n, n*0.7) 
test <- setdiff(1:n, train) 

# Training model 
g1 <- glm(default ~ balance, family="binomial", subset=train)
g2 <- glm(default ~ student, family="binomial", subset=train)
g3 <- glm(default ~ income, family="binomial", subset=train)
g4 <- glm(default ~ balance+student+income, family="binomial", subset=train)

# Testing model
miss <- NULL
for (k in 1:4) { 
  g <- get(paste("g", k, sep=""))
  # Calculate class from keyword "response" 
  pred <- predict(g, Default, type="response")[test] 
  ## yhat <- ifelse(pred > 0.5, 1, 0) 
  yhat <- rep(0, length(test)) 
  yhat[pred > 0.5] <- 1
  miss[k] <- mean(yhat != as.numeric(default[test])-1) 
}
miss 

# ======================================== [Ex] Multinomial Logistic Regression : Wine dataset ==================================== # 

# Importing library and dataset 
# install.packages('remotes')
library(remotes)
install_github("cran/rattle.data") 
library(rattle.data)
# install.packages('nnet')
library(nnet)
data(wine)

# Preview dataset wine 
str(wine)
summary(wine)
plot(wine[, -1], col=as.numeric(wine$Type) + 1)
plot(wine[, 2:7], col=as.numeric(wine$Type) + 1)
plot(wine[, 8:14], col=as.numeric(wine$Type) + 1)

# Fit model : Offers two coefficients 
fit <- multinom(Type ~ ., data=wine, trace=FALSE)
summary(fit)

# Infer coefficients with Z-test 
z <- coef(summary(fit))/summary(fit)$standard.errors
pnorm(abs(z), lower.tail=FALSE)*2 

# Calculate conditional probability 
set.seed(1)
u <- sort(sample(1:nrow(wine), 10)) 
fitted(fit)[u,]
predict(fit, wine, type="prob")[u, ]

# Predict based on probability 
prob0 <- predict(fit, wine, type="prob")
pred0 <- apply(prob0, 1, which.max)
table(pred0, wine$Type)

# Predict based on class 
pred0a <- predict(fit, wine, type="class") 
table(pred0a, wine$Type) 

pred <- predict(fit, wine, type="response")


# =================================== LDA ===================================== # 
# open the iris dataset 
data(iris)
str(iris)
summary(iris)
plot(iris[, -5], col=as.numeric(iris$Species) + 1)

# Apply LDA for iris data
library(MASS)
g <- lda(Species ~., data=iris)
plot(g)
plot(g, dimen=1)

# Compute misclassification error for training sets
pred <- predict(g)
table(pred$class, iris$Species)
mean(pred$class!=iris$Species)

# Importing library 
library(ISLR)
data(Default)
attach(Default)
library(MASS)

# Training model 
g <- lda(default~., data=Default)
pred <- predict(g, default)
table(pred$class, default)
mean(pred$class!=default)

# =========================== Thresholds =============================== #

thresholds <- seq(0, 1, 0.01) 
res <- matrix(NA, length(thresholds), 3) 

# Compute overall error, false positive, false negatives
for (i in 1:length(thresholds)) {
  decision <- rep("No", length(default))
  decision[pred$posterior[,2] >= thresholds[i]] <- "Yes"
  res[i, 1] <- mean(decision != default)
  res[i, 2] <- mean(decision[default=="No"]=="Yes")
  res[i, 3] <- mean(decision[default=="Yes"]=="No")
}

k <- 1:51
matplot(thresholds[k], res[k,], col=c(1,"orange",4), lty=c(1,4,2), type="l", xlab="Threshold", ylab="Error Rate", lwd=2)
legend("top", c("Overall Error", "False Positive", "False Negative"), col=c(1,"orange",4), lty=c(1,4,2), cex=1.2)
apply(res, 2, which.min)

# ================================== Roc curve =============================== # 

# Prerequirisite
library(ISLR)
data(Default)
attach(Default)
library(MASS)

# Train model 
g <- lda(default~., data=Default)
pred <- predict(g, default)

# Error grids
thre <- seq(0,1,0.001)
Sen <- Spe <- NULL
RES <- matrix(NA, length(thre), 4)

# Classification metrics 
colnames(RES) <- c("TP", "TN", "FP", "FN")
for (i in 1:length(thre)) {
  decision <- rep("No", length(default))
  decision[pred$posterior[,2] >= thre[i]] <- "Yes"
  Sen[i] <- mean(decision[default=="Yes"] == "Yes")
  Spe[i] <- mean(decision[default=="No"] == "No")
  RES[i,1] <- sum(decision[default=="Yes"] == "Yes")
  RES[i,2] <- sum(decision[default=="No"] == "No")
  RES[i,3] <- sum(decision=="Yes") - RES[i,1]
  RES[i,4] <- sum(default=="Yes") - RES[i,1]
}

# Visualize ROc curve 
plot(1-Spe, Sen, type="b", pch=20, xlab="False positive rate",
     col="darkblue", ylab="True positive rate", main="ROC Curve")
abline(0, 1, lty=3, col="gray")

# ====================================== Calculate AUROC =============================== # 

library(ROCR)

# Compute ROC curve
label <- factor(default, levels=c("Yes","No"),
                labels=c("TRUE","FALSE"))
preds <- prediction(pred$posterior[,2], label)
perf <- performance(preds, "tpr", "fpr" )

# Visualization 
plot(perf, lwd=4, col="darkblue")
abline(a=0, b=1, lty=2)
slotNames(perf)

k <- 1:100
list(perf@x.name, perf@x.values[[1]][k])
list(perf@y.name, perf@y.values[[1]][k])
list(perf@alpha.name, perf@alpha.values[[1]][k])

# Compute AUC
performance(preds, "auc")@y.values