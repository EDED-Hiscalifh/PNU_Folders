# Index for the Number of Clusters in Ward CA: Public Utilities
data(iris)
iristype<-iris[, 5]
#Deleting the information of clusters
irisdata <- scale(iris[, -5], scale=F)
diss_matrix <- dist(irisdata, method = "euclidean", diag = FALSE)
complete<-NbClust(irisdata, diss = diss_matrix, distance = NULL, min.nc = 2,
 max.nc = 10, method = "complete", index = "alllong")
complete
#data.frame(cluster, complete$Best.partition)
#table(cluster, complete$Best.partition)

#K-means Method
kmeans <- kmeans(irisdata, 3) # k=3 cluster solution
kmeans
table(kmeans$cluster, iristype)
