# PC Biplot for KLPGA

# Data Matrix X
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X=Data1.3.2
n <- nrow(X) 
rownames(X)
colnames(X)

Y <- scale(X,scale=T)

# Biplot based on the Singular Value Decomposition
svd.Y <- svd(Y) 
U <- svd.Y$u    
V <- svd.Y$v 
D <- diag(svd.Y$d)
G <- (sqrt(n-1)*U)[,1:2]
H <- (sqrt(1/(n-1))*V%*%D)[,1:2]
rownames(G)<-rownames(X)
rownames(H)<-colnames(X) 

# Godness-of-fit
eig <- (svd.Y$d)^2 
per <- eig/sum(eig)*100
gof <- sum(per[1:2])
round(per, 2)
round(gof, 2)

# PC Biplot
lim<-range(pretty(G))
biplot(G,H, xlab="1st PC(71.83%)",ylab="2nd PC(18.64%)", main="Biplot for KLPGA Data ",
                 xlim=lim,ylim=lim,cex=0.8,pch=16)
abline(v=0,h=0)