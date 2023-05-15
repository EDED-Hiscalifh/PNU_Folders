# PCFA Steps for KLPGA

#[Step 1] Data Matrix X
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X=Data1.3.2
rownames<-rownames(X)
p=ncol(X) 

#[Step 2] Covariance Matrix S(or Correlation Matix R)
R=round(cor(X),3)
R

#[Step 3] Spectral Decomposition 
eigen.R=eigen(R)
round(eigen.R$values, 2) # Eigenvalues
V=round(eigen.R$vectors, 2) # Eigenvectors

#[Step 4] Number of factors : m
gof=eigen.R$values/p*100 # Goodness-of fit
round(gof, 3)

#[Step 5]Factor Loadings and Communality
V2=V[,1:2]
L=V2%*%diag(sqrt(eigen.R$values[1:2]))
round(L, 3)
round(diag(L%*%t(L)), 3)

#[Step 6]Specific Variance : Psi
Psi=diag(R-L%*%t(L))
round(Psi, 3)

#{Step 7] Residual Matrix
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

# PCFA using the principal()
library(psych)
pcfa<-principal(R, nfactors=2, rotate="none")
pcfa
round(pcfa$values, 2)
gof=pcfa$values/p*100 # Goodness-of fit
round(gof, 3)
round(pcfa$residual, 2)