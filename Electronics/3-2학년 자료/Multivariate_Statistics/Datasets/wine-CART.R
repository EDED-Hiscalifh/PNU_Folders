# Classification Tree with rpart for Wine data
library(HDclassif)
data(wine)
attach(wine)
wine
# grow tree 
fit=rpart(class~., data=wine, , method="class")
fit
printcp(fit) # display the results 
summary(fit) # detailed summary of splits

# plot tree 
plot(fit, uniform=TRUE, main="Classification Tree for Wine : GINI Index")
text(fit, use.n=TRUE, all=TRUE, cex=.8)

# Confusion Table
ct=predict(fit, type="class")
confusion=table(class, ct)
APER=(1-sum(diag(prop.table(confusion))))*100
error=as.numeric(class)-predict(fit)
MSE.RSM=mean(error*error)


# grow tree 
fit1=rpart(class~., data=wine, , method="class", parms=list(split="information"))
fit1
printcp(fit1) # display the results 
summary(fit1) # detailed summary of splits

# plot tree 
plot(fit1, uniform=TRUE, main="Classification Tree for Wine : Entropy Index")
text(fit1, use.n=TRUE, all=TRUE, cex=.8)

# Confusion Table
ct1=predict(fit1, type="class")
confusion1=table(class, ct1)
APER1=(1-sum(diag(prop.table(confusion1))))*100

error=as.numeric(class)-predict(fit1)
MSE.RSM1=mean(error*error)
list(APER, APER1, MSE.RSM, MSE.RSM1)


zp <- prune(fit1, cp = 0.1)
plot(zp) #plot smaller rpart object
text(zp, use.n=TRUE, all=TRUE, cex=.8)

