# PCA Steps for KLPGA

#[Step 1] Data Matrix X
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X=Data1.3.2
rownames<-rownames(X) 

#[Step 2] Covariance Matrix S(or Correlation Matix R)
R=round(cor(X),3)
R

#[Step 3] Spectral Decomposition 
eigen.R=eigen(R)
round(eigen.R$values, 2) # Eigenvalues
V=round(eigen.R$vectors, 2) # Eigenvectors

#[Step 4] Choice of Eigenvalues and Eigenvectors
gof=eigen.R$values/sum(eigen.R$values)*100 # Goodness-of fit
round(gof, 2)

#[Step 5] PCs : liner combination of original variables 
V2=V[,1:2]
V2
#[Step 6] PCS, PCs Scores and New Data Matrix P
Z=scale(X, scale=T) # Standardized Data Matrix
Z
P=Z%*%V2            # PCs Scores
round(P, 3)

#[Step 7] Plot of PCs Scores
plot(P[,1], P[, 2], main="Plot of PCs Scores", xlab="1st PC", ylab="2nd PC")
text(P[,1], P[, 2], labels=rownames, cex=0.8, col="blue", pos=3)
abline(v=0, h=0)

#Correlations bt PCs and variables
D=diag(sqrt(eigen.R$values[1:2]))
corr=V2%*%D
round(corr, 3)

