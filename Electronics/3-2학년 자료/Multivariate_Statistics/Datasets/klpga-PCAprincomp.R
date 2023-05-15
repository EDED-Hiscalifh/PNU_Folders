setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X<-Data1.3.2[,-1]
rownames<-Data1.3.2[,1]

pca.S<-princomp(X)
summary(pca.S, loadings=T)
P<-pca.S$scores
plot(P[,1], P[, 2], main="Plot of PCs Scores", xlab="1st PC", ylab="2nd PC")
text(P[,1], P[, 2]+2, labels=rownames, cex=0.8, col="blue")
abline(v=0, h=0)

pca.R<-princomp(X, cor=T)
summary(pca.R, loadings=T)
P<-pca.R$scores
plot(P[,1], P[, 2], main="Plot of PCs Scores", xlab="1st PC", ylab="2nd PC")
text(P[,1], P[, 2], labels=rownames, cex=0.8, col="blue", pos=3)
abline(v=0, h=0)

plot(pca.S, main="Scree Plot", type="lines")