setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-Data1.3.5[, -1]

# Dissimilarity Matrix from Raw Data
rownames(X)<-Data1.3.5[, 1]
X<-scale(as.matrix(X))
m <-as.matrix(dist(X, method="euclidean", diag=T))
d<-round(m, 3)
d

# Metric MDS
con<-cmdscale(d, k=2, eig=T)
con
x<-con$points[,1]
y<-con$points[,2]
lim<-c(-max(abs(con$points)), max(abs(con$points)))
plot(x,y, xlab="Dimension 1", ylab="Dimension 2", xlim=lim, ylim=lim)
text(x,y+0.6, rownames(d), cex=0.8, pos=1)
abline(v=0, h=0)

# Non-metric MDS
library(MASS)
con<-isoMDS(d, k=2)
con
x<-con$points[,1]
y<-con$points[,2]
lim<-c(-max(abs(con$points)), max(abs(con$points)))
plot(x,y, xlab="Dimension 1", ylab="Dimension 2", xlim=lim, ylim=lim)
text(x,y+0.6, rownames(d), cex=0.8, pos=1)
abline(v=0, h=0)