# Prepare Heart dataset 
url.ht <- "https://www.statlearning.com/s/Heart.csv"
Heart <- read.csv(url.ht, h=T)
summary(Heart)
Heart <- Heart[, colnames(Heart)!="X"]
Heart[,"Sex"] <- factor(Heart[,"Sex"], 0:1, c("female", "male"))
Heart[,"Fbs"] <- factor(Heart[,"Fbs"], 0:1, c("false", "true"))
Heart[,"ExAng"] <- factor(Heart[,"ExAng"], 0:1, c("no", "yes"))
Heart[,"ChestPain"] <- as.factor(Heart[,"ChestPain"])
Heart[,"Thal"] <- as.factor(Heart[,"Thal"])
Heart[,"AHD"] <- as.factor(Heart[,"AHD"])
summary(Heart)
dim(Heart)
sum(is.na(Heart))
Heart <- na.omit(Heart)
dim(Heart)
summary(Heart)

# Separate training and test sets 
set.seed(123)
train <- sample(1:nrow(Heart), nrow(Heart)/2)
test <- setdiff(1:nrow(Heart), train)

# Training using SVMs 
library(e1071)
library(randomForest)

# SVM with a linear kernel
tune.out <- tune(svm, AHD~., data=Heart[train, ],
                 kernel="linear", ranges=list(
                   cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
heart.pred <- predict(tune.out$best.model, Heart[test,])
table(heart.pred, Heart$AHD[test])
mean(heart.pred!=Heart$AHD[test])

# SVM with a radial kernel
tune.out <- tune(svm, AHD~., data=Heart[train, ],
                 kernel="radial", ranges=list(
                   cost=c(0.1,1,10,100), gamma=c(0.5,1,2,3)))
heart.pred <- predict(tune.out$best.model, Heart[test,])
table(heart.pred, Heart$AHD[test])
mean(heart.pred!=Heart$AHD[test])

# SVM with a polynomial kernel
tune.out <- tune(svm, AHD~.,data=Heart[train, ],
                 kernel="polynomial", ranges=list(
                   cost=c(0.1,1,10,100), degree=c(1,2,3)))
heart.pred <- predict(tune.out$best.model, Heart[test,])
table(heart.pred, Heart$AHD[test])
mean(heart.pred!=Heart$AHD[test])

# SVM with a sigmoid kernel
tune.out <- tune(svm, AHD~.,data=Heart[train, ],
                 kernel="sigmoid", ranges=list(
                   cost=c(0.1,1,10,100), gamma=c(0.5,1,2,3)))
heart.pred <- predict(tune.out$best.model, Heart[test,])
table(heart.pred, Heart$AHD[test])
mean(heart.pred!=Heart$AHD[test])

# Simulation study using differernt kernels of 20 replications 
set.seed(123)
N <- 20
Err <- matrix(0, N, 5)

for (i in 1:N) {
  train <- sample(1:nrow(Heart), floor(nrow(Heart)*2/3))
  test <- setdiff(1:nrow(Heart), train)
  g1 <- randomForest(x=Heart[train,-14], y=Heart[train,14],
                     xtest=Heart[test,-14], ytest=Heart[test,14], mtry=4)
  Err[i,1] <- g1$test$err.rate[500,1]
  g2 <- tune(svm, AHD~., data=Heart[train, ], kernel="linear",
             ranges=list(cost=c(0.001, 0.01, 0.1, 1, 5, 10, 100)))
  p2 <- predict(g2$best.model, Heart[test,])
  Err[i,2] <- mean(p2!=Heart$AHD[test])
  g3 <- tune(svm, AHD~., data=Heart[train, ], kernel="radial",
             ranges=list(cost=c(0.1,1,10,100), gamma=c(0.5,1,2,3)))
  p3 <- predict(g3$best.model, Heart[test,])
  Err[i,3] <- mean(p3!=Heart$AHD[test])
  
  g4 <- tune(svm, AHD~.,data=Heart[train, ],kernel="polynomial",
             ranges=list(cost=c(0.1,1,10,100), degree=c(1,2,3)))
  p4 <- predict(g4$best.model, Heart[test,])
  Err[i,4] <- mean(p4!=Heart$AHD[test])
  g5 <- tune(svm, AHD~.,data=Heart[train, ],kernel="sigmoid",
             ranges=list(cost=c(0.1,1,10,100), gamma=c(0.5,1,2,3)))
  p5 <- predict(g5$best.model, Heart[test,])
  Err[i,5] <- mean(p5!=Heart$AHD[test])
}

# Visualize results 
labels <- c("RF","SVM.linear","SVM.radial","SVM.poly","SVM.sig")
boxplot(Err, boxwex=0.5, main="Random Forest and SVM", col=2:6,
        names=labels, ylab="Classification Error Rates",
        ylim=c(0,0.4))
colnames(Err) <- labels
apply(Err, 2, summary)