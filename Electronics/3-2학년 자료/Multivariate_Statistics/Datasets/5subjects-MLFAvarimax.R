# MLFA Steps for 5 Subjects

# Data Matrix X
library("MVT")
data(examScor)
X=examScor
p=ncol(X) 

# Covariance Matrix S(or Correlation Matix R)
R=round(cor(X),3)
R

# ML Estimation using the factanal( )
library(psych)
mlfa<-factanal(covmat=R, factors = 2, rotation="varimax")
mlfa

# Residual Matrix
L=mlfa$loading[, 1:2]
Psi=mlfa$uniquenesses
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

# Factor Loadings Plot
lim<-c(-0.1, 1)
plot(L[,1], L[,2],main="Plot of Factor Loadings : Varimax",  xlab="f1", ylab="f2",
                       xlim=lim, ylim=lim)
text(L[,1], L[, 2], labels=rownames(L), cex=0.8, col="blue", pos=1)
abline(v=0, h=0)
arrows(0,0, L[,1], L[, 2], col=2, code=2, length=0.1)