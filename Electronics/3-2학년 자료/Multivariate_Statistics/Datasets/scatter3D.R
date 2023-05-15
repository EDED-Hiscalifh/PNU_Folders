setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.1.1<-read.table("3subjects.txt", header=T)
X<-as.matrix(Data1.1.1[, -1])
rownames(X)<-Data1.1.1[,1]
colnames(X)
library(rgl)

# Observations in Variables Space
lim<-c(0, 100)
plot3d(X[,1], X[,2], X[,3],xlim=lim, ylim=lim, zlim=lim,
xlab="Mechanics", ylab="Algebra", zlab="Statistics")
text3d(X[,1], X[,2], X[,3],rownames(X))

# Variables in Observations Space
plot3d(X[1,], X[2,], X[3,], xlim=lim, ylim=lim, zlim=lim,
xlab="Student1", ylab="Student2", zlab="Student3")
text3d(X[1,], X[2,], X[3,], colnames(X))