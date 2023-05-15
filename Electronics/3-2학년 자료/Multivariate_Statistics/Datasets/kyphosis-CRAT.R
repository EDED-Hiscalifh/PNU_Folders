# Classification Tree with rpart for kyphosis data
library(rpart)
data(kyphosis)
attach(kyphosis)
# Grow tree 
fit <- rpart(Kyphosis ~ Age + Number + Start,
    method="class", data=kyphosis)
printcp(fit) # display the results 
summary(fit) # detailed summary of splits

 # Plot tree 
 plot(fit, uniform=TRUE, 
    main="Classification Tree for Kyphosis")
 text(fit, use.n=TRUE, all=TRUE, cex=.8)

# Confusion Table
ct=predict(fit, type="class")
confusion=table(Kyphosis, ct)
APER=(1-sum(diag(prop.table(confusion))))*100
error=as.numeric(Kyphosis)-predict(fit)
MSE.RSM=mean(error*error)
list(confusion, APER, MSE.RSM)

# Prune Tree
#subtree <- prune(fit, cp=fit$cptable[which.min(fit$cptable[,"xerror"]),"CP"])
subtree <- prune(fit,cp=0.019608)
plot(subtree, uniform=TRUE, main="Classification Tree for Kyphosis : Pruning Tree")
text(subtree, use.n=TRUE, all=TRUE, cex=.8)

# Confusion Table for Pruned Tree
ct1=predict(subtree, type="class")
confusion1=table(Kyphosis, ct1)
APER1=(1-sum(diag(prop.table(confusion1))))*100
error1=as.numeric(Kyphosis)-predict(subtree)
MSE.RSM1=mean(error*error)
list(confusion1, APER1, MSE.RSM1)
