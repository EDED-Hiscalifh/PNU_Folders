# PCA steps and correlation coefficients bt PCs and variables

#[Step 1] Input Data 
S1=matrix(c(80,44,44,80), byrow=T, nrow=2)
S2=matrix(c(8000,440,440,80), byrow=T, nrow=2)

#[Step 2] Covariance Matrix S(or Correlation Matix R)
S=S1 #S=S2
S

#[Step 3] Spectral Decomposition 
eigen.S=eigen(S)
round(eigen.S$values, 3) # Eigenvalues
V=round(eigen.S$vectors, 3) # Eigenvaectors
V
#[Step 4] Choice of Eigenvalues and Eigenvectors
gof=eigen.S$values/sum(eigen.S$values)*100 # Goodness-of-fit
round(gof, 2)

#[Step 5] PCs : linear combination of original variables 
V2=V[,1:2]
V2

#Correlations bt PCs and variables
Ds=diag(diag(1/sqrt(S)))
D=diag(sqrt(eigen.S$values))
corr=Ds%*%V2%*%D
corr
