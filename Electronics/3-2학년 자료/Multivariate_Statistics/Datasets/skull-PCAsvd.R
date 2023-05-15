# PCA Steps based on the SVD for Skull Data

#[Step 1] Data Matrix X
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("skull.txt", header=T)
Z=as.matrix(Data1.3.2)
rownames<-rownames(Z) 
colnames<-colnames(Z)
n=nrow(Z)

#[Step 2] Singular Values Decomposition 
svd.Z=svd(Z)
U=svd.Z$u # Right singular vectors 
V=svd.Z$v # Left singular vectors : Eigenvectors
round(V, 2)
D=diag(svd.Z$d)
#[Step 3] Choice of Singular Valuies and Eigenvectors
round(svd.Z$d, 2)
eigen=(svd.Z$d)^2
round(eigen/(n-1), 2)
gof=eigen/sum(eigen)*100 # Goodness-of fit
round(gof, 2)

#[Step 5] PCs : liner combination of original variables 
V3=V[,1:3]
V3
round(t(V3), 2)
#[Step 6] PCS, PCs Scores and New Data Matrix P
Z # Standardized Data Matrix
#P=U%*%D            # PCs Scores : P=Z%*%V3
round(P, 3)

#[Step 7] Plot of PCs Scores
par(mfrow=c(2,2))
plot(P[,1], P[, 2], main="(a) Plot of PCs Scores", xlab="1st PC", ylab="2nd PC")
text(P[,1], P[, 2], labels=rownames, cex=0.8, col="blue", pos=1)
abline(v=0, h=0)

plot(P[,1], P[, 3], main="(b) Plot of PCs Scores", xlab="1st PC", ylab="3rd PC")
text(P[,1], P[, 3], labels=rownames, cex=0.8, col="blue", pos=1)
abline(v=0, h=0)

plot(P[,2], P[, 3], main="(c) Plot of PCs Scores", xlab="2nd PC", ylab="3rd PC")
text(P[,2], P[, 3], labels=rownames, cex=0.8, col="blue", pos=1)
abline(v=0, h=0)



#Correlations bt PCs and variables
D=diag(svd.Z$d[1:3]/sqrt(n-1))
corr=V3%*%D
round(corr, 3)
