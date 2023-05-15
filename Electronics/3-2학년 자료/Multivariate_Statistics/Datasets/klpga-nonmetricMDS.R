setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X<-scale(Data1.3.2, scale=T)
X<-as.matrix(Data1.3.2)
선수<-rownames(X) 
n<-nrow(X)
p<-ncol(X)


# 표준화 유클리드거리
D <- as.matrix(dist(X, method="euclidean"))

# Nonmetric MDS
win.graph()
library(MASS)
con<-isoMDS(D, k=2)
con
x<-con$points[,1]
y<-con$points[,2]
lim1<-c(-max(abs(x)), max(abs(x)))
lim2<-c(-max(abs(y)), max(abs(y)))
plot(x,y, xlab="Dim1", ylab="Dim2", xlim=lim1, ylim=lim2)
text(x,y,선수, cex=0.8, pos=2)
abline(v=0, h=0)

# Shepard  Diagram
win.graph()
dist_sh <- Shepard(D[lower.tri(D)], con$points)
dist_sh
cdist_sh=cbind(dist_sh$x, dist_sh$y, dist_sh$yf)
cdist_sh
plot(cdist_sh[,1], cdist_sh[,3], pch = ".", xlab = "Dissimilarity", ylab = "Distance", 
xlim = range(cdist_sh[,1]), ylim = range(cdist_sh[,1]))
lines(cdist_sh[,1],  cdist_sh[,3], type = "S")

# Image Diagram
win.graph()
plot(cdist_sh[,2], cdist_sh[,3], pch = ".", xlab = "FitDissimilarity", ylab = "Distance", 
xlim = range(cdist_sh[,2]), ylim = range(cdist_sh[,2]))
lines(cdist_sh[,2],  cdist_sh[,3], type = "p")
