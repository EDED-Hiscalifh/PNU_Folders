library(MASS)
library(shapes)
par(mfrow=c(1,2))
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
C<- as.matrix(read.table("morse.txt", header=T, check.names=F))
C 
n<-nrow(C)
# Standard Transformation : cij(similarity) to dij(dissimilarity)
J<-matrix(1,n,n)
cii=diag(diag(C))%*%J
cij=C
cjj=J%*%diag(diag(C))
d<-sqrt(cii-2*cij+cjj)
d
# Metric MDS
con<- cmdscale(d, k=2, eig=T)
con
x<-con$points[,1]
y<-con$points[,2]
lim<-c(-max(abs(con$points)), max(abs(con$points)))
plot(x,y, xlab="Dimension 1", ylab="Dimension 2", xlim=lim, ylim=lim, main="Metric MDS")
text(x,y+0.6, colnames(d), cex=0.8)
abline(v=0, h=0)
A<-con$points
A
# Nonmetric MDS
con<- isoMDS(d, k=2)
con
x<-con$points[,1]
y<-con$points[,2]
lim<-c(-max(abs(con$points)), max(abs(con$points)))
plot(x,y, xlab="Dimension 1", ylab="Dimension 2", xlim=lim, ylim=lim, main="Nonmetric MDS")
text(x,y+0.6, colnames(d), cex=0.8)
abline(v=0, h=0)
B<-con$points

# OPA of Shape Analysis
PBA<-procOPA(A,B) # B onto A
PBA
plotshapes(A,B,joinline=c(1:10, 1))
plotshapes(PBA$Bhat,joinline=c(1:10, 1))
PAB<-procOPA(B,A) # A onto B
PAB
plotshapes(PBA$Bhat, PAB$Bhat,joinline=c(1:10, 1))

# Algorithms for OPA(A, B)
A
B
J<-matrix(1,n,n)
I<-diag(diag(J))
H<-I-J/n               
Ac<-H%*%A 
Bc<-H%*%B 
t(Ac)%*%Bc
svd<-svd(t(Ac)%*%Bc) 
svd
gamma<-svd$v%*%t(svd$u)
beta<-sum(diag((t(Ac)%*%Bc%*%gamma)))/sum(diag(t(Bc)%*%Bc))
OPSS<-sum(diag((t(Ac)%*%Ac))) + beta^2%*% sum(diag((t(Bc)%*%Bc))) - 2%*%beta%*%sum(diag(svd$d))
RMSD<-sqrt(OPSS/n)
list(beta, gamma, OPSS, RMSD)

