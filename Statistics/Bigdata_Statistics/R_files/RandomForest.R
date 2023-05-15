# ============================== Package for tree methods ==============================
library(tree)
library(randomForest)
library(ISLR) 
library(rpart)
library(rpart.plot)
library(MASS)
library(leaps)

# ============================== Importing Datasets ==============================
data(Boston)
data(Hitters)
url.ht <- "https://www.statlearning.com/s/Heart.csv"
Heart <- read.csv(url.ht, h=T)

Heart <- Heart[, colnames(Heart)!="X"]
Heart[,"Sex"] <- factor(Heart[,"Sex"], 0:1, c("female", "male"))
Heart[,"Fbs"] <- factor(Heart[,"Fbs"], 0:1, c("false", "true"))
Heart[,"ExAng"] <- factor(Heart[,"ExAng"], 0:1, c("no", "yes"))
Heart[,"ChestPain"] <- as.factor(Heart[,"ChestPain"])
Heart[,"Thal"] <- as.factor(Heart[,"Thal"])
Heart[,"AHD"] <- as.factor(Heart[,"AHD"])
Heart <- na.omit(Heart)

# ============================== Random Forest ==============================

# Train-test split
set.seed(123)
train <- sample(1:nrow(Heart), nrow(Heart)/2) 
test <- setdiff(1:nrow(Heart), train) 
                
# Bagging(m=13) 
bag.heart <- randomForest(x=Heart[train, -14], y=Heart[train,14], 
                          xtest=Heart[test, -14], ytest=Heart[test,14], 
                          mtry=13, importance=TRUE) 
bag.heart
bag.conf <- bag.heart$test$confusion[1:2,1:2] 

# Miss classification Error of m = 13
1 - sum(diag(bag.conf))/sum(bag.conf)

# Bagging(m=1) 
rf1 <- randomForest(x=Heart[train,-14], y=Heart[train,14], 
                    xtest=Heart[test,-14], ytest=Heart[test,14], 
                    mtry=1, importance=TRUE) 
rf1.conf <- rf1$test$confusion[1:2, 1:2]
1- sum(diag(rf1.conf))/sum(rf1.conf)

## Random forest with m=4
rf2 <- randomForest(x=Heart[train,-14], y=Heart[train,14],
                    xtest=Heart[test,-14], ytest=Heart[test,14],
                    mtry=4, importance=TRUE)
rf2.conf <- rf2$test$confusion[1:2,1:2]
1- sum(diag(rf2.conf))/sum(rf2.conf)

## Random forest with m=6
rf3 <- randomForest(x=Heart[train,-14], y=Heart[train,14],
                    xtest=Heart[test,-14], ytest=Heart[test,14],
                    mtry=6, importance=TRUE)
rf3.conf <- rf3$test$confusion[1:2,1:2]
1- sum(diag(rf3.conf))/sum(rf3.conf)

# Scatter plot of feature importance 
varImpPlot(rf1)
varImpPlot(rf2)
varImpPlot(rf3)

# Horizontal bar plot of feature importance 
(imp1 <- importance(rf1))
(imp2 <- importance(rf2))
(imp3 <- importance(rf3))

par(mfrow=c(1,3))
# Based on MeanDecreaseAccuracy
barplot(sort(imp1[,3]), main="RF (m=1)", horiz=TRUE, col=2)
barplot(sort(imp2[,3]), main="RF (m=4)", horiz=TRUE, col=2)
barplot(sort(imp3[,3]), main="RF (m=6)", horiz=TRUE, col=2)

# Based on MeanDecreaseGini
barplot(sort(imp1[,4]), main="RF (m=1)", horiz=TRUE, col=2)
barplot(sort(imp2[,4]), main="RF (m=4)", horiz=TRUE, col=2)
barplot(sort(imp3[,4]), main="RF (m=6)", horiz=TRUE, col=2)

dev.off() 

# ============================== Boosting ==============================

# Prerequirisite : Importing library, dataset, preprocessing 
library(gbm)

# Train-Test split 
set.seed(123)
train <- sample(1:nrow(Heart), nrow(Heart)/2)
test <- setdiff(1:nrow(Heart), train)

# Create (0,1) response
Heart0 <- Heart
Heart0[,"AHD"] <- as.numeric(Heart$AHD)-1
# boosting (d=1)
boost.d1 <- gbm(AHD~., data=Heart0[train, ], n.trees=1000,
                distribution="bernoulli", interaction.depth=1)

# Results of feature selection 
summary(boost.d1)

# Calcualte missclassification error 
# We need to set value of keyword n.trees for boosting trees 
yhat.d1 <- predict(boost.d1, newdata=Heart0[test, ], type="response", n.trees=1000)
phat.d1 <- rep(0, length(yhat.d1))
phat.d1[yhat.d1 > 0.5] <- 1
mean(phat.d1!=Heart0[test, "AHD"])

# boosting (d=2)
# Training model with max_depth = 2
boost.d2 <- gbm(AHD~., data=Heart0[train, ], n.trees=1000,
                distribution="bernoulli", interaction.depth=2)

# Make predictions 
yhat.d2 <- predict(boost.d2, newdata=Heart0[test, ],
                   type="response", n.trees=1000)
phat.d2 <- rep(0, length(yhat.d2))
phat.d2[yhat.d2 > 0.5] <- 1
mean(phat.d2!=Heart0[test, "AHD"])

# boosting (d=3)
# Training model with max_depth = 3
boost.d3 <- gbm(AHD~., data=Heart0[train, ], n.trees=1000,
                distribution="bernoulli", interaction.depth=3)

# Make predictions
yhat.d3 <- predict(boost.d3, newdata=Heart0[test, ],
                   type="response", n.trees=1000)
phat.d3 <- rep(0, length(yhat.d3))
phat.d3[yhat.d3 > 0.5] <- 1
mean(phat.d3!=Heart0[test, "AHD"])

# boosting (d=4)
# Training model with max_depth = 4
boost.d4 <- gbm(AHD~., data=Heart0[train, ], n.trees=1000,
                distribution="bernoulli", interaction.depth=4)

# Make predictions 
yhat.d4 <- predict(boost.d4, newdata=Heart0[test, ],
                   type="response", n.trees=1000)
phat.d4 <- rep(0, length(yhat.d4))
phat.d4[yhat.d4 > 0.5] <- 1
mean(phat.d4!=Heart0[test, "AHD"])

# Simulation: Boosting with d=1, 2, 3 and 4
# The number of trees: 1 to 3000

# Set grids 
set.seed(1111)
Err <- matrix(0, 3000, 4)

# Training models with n.trees, interaction.depth : 1 to 4
for (k in 1:4) {
  boost <- gbm(AHD~., data=Heart0[train, ], n.trees=3000,
               distribution="bernoulli", interaction.depth=k)
  for (i in 1:3000) {
    # Make predictions of n.trees : 1 to 3000 
    yhat <- predict(boost, newdata=Heart0[test, ],
                    type="response", n.trees=i)
    phat <- rep(0, length(yhat))
    phat[yhat > 0.5] <- 1
    Err[i,k] <- mean(phat!=Heart0[test, "AHD"])
  }
}

# Visualize results 
labels <- c("d = 1", "d = 2", "d = 3", "d = 4")
matplot(Err, type="l", xlab="Number of Trees", lty=2, col=1:4,
        ylab="Classification Error Rate")
legend("topright", legend=labels, col=1:4, lty=1)

# View statistical reports 
colnames(Err) <- labels
apply(Err, 2, summary)
apply(Err[-c(1:100),], 2, summary)

install.packages('BART')

# ============================== BART ==============================

# Prerequirisite 
library(BART)

# Train-test split 
set.seed(123)
train <- sample(1:nrow(Heart), nrow(Heart)/2)
test <- setdiff(1:nrow(Heart), train)
x <- Heart[, -14]
y <- as.numeric(Heart[, 14])-1
xtrain <- x[train, ]
ytrain <- y[train]
xtest <- x[-train, ]
ytest <- y[-train]

# Logistic BART 
set.seed(11)
fit1 <- lbart(xtrain, ytrain, x.test=xtest)
names(fit1)

# Make predictions 
prob1 <- rep(0, length(ytest))
prob1[fit1$prob.test.mean > 0.5] <- 1
mean(prob1!=ytest)

# Probit BART 
set.seed(22)
fit2 <- pbart(xtrain, ytrain, x.test=xtest)

# Make Prediction 
prob2 <- rep(0, length(ytest))
prob2[fit2$prob.test.mean > 0.5] <- 1
mean(prob2!=ytest)

# Visualize results : lbart ~ pbart 
cbind(fit1$prob.test.mean, fit2$prob.test.mean)
plot(fit1$prob.test.mean, fit2$prob.test.mean, col=ytest+2,
     xlab="Logistic BART", ylab="Probit BART")
abline(0, 1, lty=3, col="grey")
abline(v=0.5, lty=1, col="grey")
abline(h=0.5, lty=1, col="grey")
legend("topleft", col=c(2,3), pch=1,
       legend=c("AHD = No", "AHD = Yes"))

# Revisit Boston data set with a quantitative response
library(MASS)
summary(Boston)
dim(Boston)

# Train-test split
set.seed(111)
train <- sample(1:nrow(Boston), floor(nrow(Boston)*2/3))
boston.test <- Boston[-train, "medv"]

# Calculate misssclassification error rate of Regression tree
library(tree)
tree.boston <- tree(medv ~ ., Boston, subset=train)
yhat <- predict(tree.boston, newdata=Boston[-train, ])
mean((yhat - boston.test)^2)

# Calculate missclassification error rate of LSE: least square estimates
g0 <- lm(medv ~ ., Boston, subset=train)
pred0 <- predict(g0, Boston[-train,])
mean((pred0 - boston.test)^2)

# Calculate missclassification error rate of Bagging
library(randomForest)
g1 <- randomForest(medv ~ ., data=Boston, mtry=13, subset=train)
yhat1 <- predict(g1, newdata=Boston[-train, ])
mean((yhat1 - boston.test)^2)

# Calculate missclassification error rate of Random Forest (m = 4)
g2 <- randomForest(medv ~ ., data=Boston, mtry=4, subset=train)
yhat2 <- predict(g2, newdata=Boston[-train, ])
mean((yhat2 - boston.test)^2)

# Calculate missclassification error rate of Boosting (d = 4)
library(gbm)
g3 <- gbm(medv~., data = Boston[train, ], distribution="gaussian", n.trees=5000, interaction.depth=4)
yhat3 <- predict(g3, newdata=Boston[-train, ], n.trees=5000)
mean((yhat3 - boston.test)^2)

# Calculate missclassifcation error rate of BART
library(BART)
g4 <- gbart(Boston[train, 1:13], Boston[train, "medv"], x.test=Boston[-train, 1:13])
yhat4 <- g4$yhat.test.mean
mean((yhat4 - boston.test)^2)

# Simulation: 4 ensemble methods
set.seed(1111)
N <- 20
ERR <- matrix(0, N, 4)

# replicate 20 times 
for (i in 1:N) {
  train <- sample(1:nrow(Boston), floor(nrow(Boston)*2/3))
  boston.test <- Boston[-train, "medv"]
  
  # Bagging
  g1 <- randomForest(medv ~ ., data=Boston, mtry=13, subset=train)
  yhat1 <- predict(g1, newdata=Boston[-train, ])
  ERR[i,1] <- mean((yhat1 - boston.test)^2)
  
  # Random forest
  g2 <- randomForest(medv ~ ., data=Boston, mtry=4, subset=train)
  yhat2 <- predict(g2, newdata=Boston[-train, ])
  ERR[i, 2] <- mean((yhat2 - boston.test)^2)
  
  # Boosting
  g3 <- gbm(medv~., data = Boston[train, ], n.trees=5000, distribution="gaussian", interaction.depth=4)
  yhat3 <- predict(g3, newdata=Boston[-train, ], n.trees=5000)
  ERR[i, 3] <- mean((yhat3 - boston.test)^2)
  
  # BART
  invisible(capture.output(g4 <- gbart(Boston[train, 1:13], Boston[train, "medv"], x.test=Boston[-train, 1:13])))
  yhat4 <- g4$yhat.test.mean
  ERR[i, 4] <- mean((yhat4 - boston.test)^2)
}

# Visualize simulation results 
labels <- c("Bagging", "RF", "Boosting", "BART")
boxplot(ERR, boxwex=0.5, main="Ensemble Methods", col=2:5, names=labels, ylab="Mean Squared Errors", ylim=c(0,30))
colnames(ERR) <- labels

# Check statistical reports 
apply(ERR, 2, summary)
apply(ERR, 2, var)

# Check rankings 
RA <- t(apply(ERR, 1, rank))
RA
apply(RA, 2, table)