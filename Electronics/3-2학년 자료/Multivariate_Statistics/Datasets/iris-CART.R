# Classification Tree with rpart for Iris 
library(rpart)
# Iris flower data : setosa, versicolor, virginica
data(iris)
attach(iris)

# grow tree 
fit=rpart(Species~., data=iris, method="class")
fit
printcp(fit) # display the results 
summary(fit) # detailed summary of splits
# RSM : mean squared error 
error=as.numeric(Species)-predict(fit)
MSE.RSM=mean(error*error)

# plot tree 
plot(fit, uniform=TRUE, main="Classification Tree for Iris with Gini Index")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# Confusion Table
ct=predict(fit, type="class")
table(Species, ct)

# CVM :  mean squared error 
library(caret)
set.seed(1000) #reproducability setting
subsample<-createDataPartition(y=iris$Species, p=0.7, list=FALSE) 
iris.train<-iris[subsample, ]
iris.test<-iris[-subsample, ]
# grow tree 
testfit=rpart(Species~., data=iris.train, method="class")
error=as.numeric(iris.test$Species)-predict(testfit, newdata=iris.test)
MSE.CV=mean(error*error)
list(MSE.RSM, MSE.CV)


