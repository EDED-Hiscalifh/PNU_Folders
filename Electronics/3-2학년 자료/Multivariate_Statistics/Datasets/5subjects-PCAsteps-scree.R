# Steps for PCA

#[Step 1] Data Matrix X
library("MVT")
data(examScor)
X=examScor

#[Step 2] Covariance Matrix S(or Correlation Matix R)
S=round(cov(X),3)
S

#[Step 3] Spectrla Decompositoin 
eigen.S=eigen(S)
round(eigen.S$values, 3) # Eigenvalus
V=round(eigen.S$vectors, 3) # Eigenvaectors
V
#[Step 4] Choice of Eigenvalues and Eigenvectors
gof=eigen.S$values/sum(eigen.S$values)*100 # Goodness-of fit
round(gof, 2)
plot(eigen.S$values, type="b", main="Scree Graph", xlab="Component Number", ylab="Eigenvalue")

#[Step 5] PCs : liner combination of original variables 
V2=V[,1:2]
V2
#[Step 6] PCS, PCs Scores and New Data Matrix P
Y=scale(X, scale=F) # Centred Data Matrix
P=Y%*%V2            # PCs Scores
P

#[Step 7] Plot of PCs Scores
plot(P[,1], P[, 2], main="Plot of PCs Scores", xlab="1st PC", ylab="2nd PC")
text(P[,1], P[, 2]+2, labels=rownames(P), cex=0.8, col="blue")
abline(v=0, h=0)
