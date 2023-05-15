setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.3<-read.table("protein.txt", header=T)
X<-Data1.3.3[,-(1:2)]
rownames(X)<-Data1.3.3[,2]
colnames(X)

# Biplot Analysis
n <- nrow(X) 
p <- ncol(X)

Y <- scale(X,scale=F)

svd.Y <- svd(Y)
U <- svd.Y$u 
V <- svd.Y$v 
D <- diag(svd.Y$d)
A <- (sqrt(n-1)*U)[,1:2]
B <- (sqrt(1/(n-1))*V%*%D)[,1:2]
rownames(A) <- rownames(X)
rownames(B) <- colnames(X)


eig <- (svd.Y$d)^2 
per <- eig/sum(eig)*100
gof <- sum(per[1:2])
per
gof

par(pty="s")
lim <- range(pretty(B))
plot(B[,1],B[,2],xlab="Dim1",ylab="Dim2",xlim=lim,ylim=lim,pch=15,col=2)
abline(v=0,h=0)
text(B[,1],B[,2],rownames(B),cex=0.8,col=2,pos=3)
arrows(0,0,B[,1],B[,2],col=2,code=0)

lim2 <- range(pretty(A))
plot(A[,1],A[,2],xlab="Dim1",ylab="Dim2",xlim=lim2,ylim=lim2,pch=16)
abline(v=0,h=0)
text(A[,1],A[,2],rownames(A),cex=0.8,pos=3)
