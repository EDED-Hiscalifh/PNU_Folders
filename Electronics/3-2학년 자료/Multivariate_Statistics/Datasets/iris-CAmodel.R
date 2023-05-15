# CA based on Statistical Models: Iris flowers 
data(iris)
iristype<-iris[, 5]
#Deleting the information of clusters
irisdata<-iris[, -5]

# Model based clustering 
library(mclust)
irismc<-Mclust(irisdata, G=3, modelNames=c("EII", "VII"))
irismc
attributes(irismc)

rismc$BIC # BIC criterion
irismc$parameters # Estimated parameters 
irismc$classification # Clusters

table(irismc$classification, iristype)
plot(irismc,what="classification")
