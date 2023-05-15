# Index for the Number of Clusters in K-Means CA: Public Utilities
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data5.3.1<-read.table("utility.txt", header=T)
X<-Data5.3.1[,-1]
Z<-scale(X)
company=Data5.3.1[,1]

library("NbClust")
#CCC Index
ccc<-NbClust(Z, distance="euclidean", min.nc = 2, max.nc = 8,
 method = "kmeans", index = "ccc")
ccc
plot(2:8, type="b", ccc$All.index, xlab="Number of Clusters",
   ylab="CCC")

#Dindex Index
dindex<-NbClust(Z, distance="euclidean", min.nc = 2, max.nc = 8,
 method = "kmeans", index = "dindex")
dindex

#All Indices
allindex<-NbClust(Z, distance="euclidean", min.nc = 2, max.nc = 8, 
method = "kmeans", index = "all", )
allindex