setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Table1.1<-read.table("T1-1.txt", header=T)
Table1.1
X<-as.matrix(Table1.1[, -1])
X
rownames(X)<-Table1.1[,1]
colnames(X)
library(rgl)
library(plot3D)
plot3d(X[1,], X[2,], X[3,],xlim=c(0, 100), ylim=c(0, 100), zlim=c(0, 100),
xlab="Student1", ylab="Student2", zlab="Student3")
text3d(X[1,], X[2,], X[3,],colnames(X))
library(plot3D)

x0<-c(0,0,0)
y0<-c(0,0,0)
z0<-c(0,0,0)

scatter3D(X[1,], X[2,], X[3,], colvar = NULL, pch=19, ticktype = "detailed")
arrows3D(x0, y0, z0, X[1,], X[2,], X[3,], 
xlim=c(0, 100), ylim=c(0, 100), zlim=c(0, 100), 
xlab="Student1", ylab="Student2", zlab="Student3")
text3D(X[1,], X[2,], X[3,], add=T,  c("Mechanics", "Algebra", "Statistics"))

