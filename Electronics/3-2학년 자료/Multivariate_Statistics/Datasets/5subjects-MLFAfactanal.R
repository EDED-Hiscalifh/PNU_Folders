# MLFA Steps for 5 Subjects
# Data Matrix X
library("MVT")
data(examScor)
X=examScor
Z<-scale(X, scale=T)
p=ncol(X) 

# Covariance Matrix S(or Correlation Matix R)
R=round(cor(X),3)
R

# ML Estimation using the factanal( )
library(psych)
mlfa<-factanal(Z, factors = 2, rotation="varimax")
mlfa

# Residual Matrix
L=mlfa$loading[, 1:2]
Psi=mlfa$uniquenesses
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)