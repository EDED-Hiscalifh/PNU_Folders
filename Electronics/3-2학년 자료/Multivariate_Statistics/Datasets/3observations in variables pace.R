Table1.1<-read.table("c:/R과 함께하는 다변량자료분석/R_code_data/T1-1.txt", header=T)
Table1.1
X<-as.matrix(Table1.1[, -1])
rownames(X)<-Table1.1[,1]
colnames(X)
library(rgl)
plot3d(X[,1], X[,2], X[,3],xlim=c(0, 100), ylim=c(0, 100), zlim=c(0, 100),
xlab="Mechanics", ylab="Algebra", zlab="Statistics")
text3d(X[,1], X[,2], X[,3],rownames(X))




