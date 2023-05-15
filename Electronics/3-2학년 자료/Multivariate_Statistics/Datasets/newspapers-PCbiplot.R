setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
data9.3.1<-read.table("newspapers.txt", header=T)
X<-data9.3.1
round(cor(X), 2)

Y<-scale(X)
rownames(Y)<-rownames(X)
colnames(Y)<-colnames(X)

# Biplot based on the Singular Value Decomposition
svd.Y <- svd(Y) 
U <- svd.Y$u    
V <- svd.Y$v 
D <- diag(svd.Y$d)
A <-  (sqrt(n-1)*U)[,1:2]
B <- (sqrt(1/(n-1))*V%*%D)[,1:2] 

rownames(A)<-rownames(X)
rownames(B)<-colnames(X)
# Godness-of-fit
eig <- (svd.Y$d)^2 
per <- eig/sum(eig)*100
gof <- sum(per[1:2])
list(per, gof)

win.graph()
# PC Biplot
lim<-range(pretty(A))
biplot(A,B, xlab="Dim1", ylab="Dim2", main=" (a) PC Biplot: SVD",
                 xlim=lim,ylim=lim, cex=1.0, pch=16, col=1)
abline(v=0,h=0)

pcasvd<-prcomp(Y, scale=F)
summary(pcasvd)
biplot(pcasvd, pc.biplot=T, xlab="Dim1", ylab="Dim2", 
             main="(b) PC Biplot : pc.biplot() ", cex=1.0, pch=16)
abline(v=0,h=0)



