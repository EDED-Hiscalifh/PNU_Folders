# PCA using R function princomp( )

library("MVT")
data(examScor)
X=examScor
pca.S<-princomp(X)
summary(pca.S, loadings=T)
P<-pca.S$scores


plot(P[,1], P[, 2], main="Plot of PCs Scores", xlab="1st PC", ylab="2nd PC")
text(P[,1], P[, 2]+2, labels=rownames(P), cex=0.8, col="blue")
abline(v=0, h=0)

biplot(pca.S)
abline(v=0, h=0)


