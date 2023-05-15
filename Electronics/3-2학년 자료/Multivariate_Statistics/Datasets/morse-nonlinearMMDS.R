#Morse data의 비선형 계량형 MDS
# Dissimilarity Matrix from Similarity Matrix
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data7.2.3<-read.table("morse.txt", header=T, check.names=F)
C<-as.matrix(Data7.2.3)
부호=colnames(C)
n<-nrow(C)

# Standard Transformation : cij(similarity) to dij(dissimilarity)
J<-matrix(1,n,n)
cii=diag(diag(C))%*%J
cij=C
cjj=J%*%diag(diag(C))
D<-sqrt(cii-2*cij+cjj)
D
# Nonlinear Metric MDS
win.graph()
library(MASS)
con<-sammon(D, k=2, magic=0.3)
con
x<-con$points[,1]
y<-con$points[,2]
lim1<-c(-max(abs(x)), max(abs(x)))
lim2<-c(-max(abs(y)), max(abs(y)))

plot(x,y, xlab="Dim1", ylab="Dim2", xlim=lim1, ylim=lim2)
text(x,y,부호, cex=0.8, pos=3)
abline(v=0, h=0)