setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data2.8.2<-read.table("airpollution.txt", header=T)
X=Data2.8.2
rownames<-rownames(X)
p=ncol(X) 
n=nrow(X)
X
Z<-scale(X, scale=T)

# Covariance Matrix S(or Correlation Matix R)
R=round(cor(X),3)
R

# ML Estimation using the factanal( )
library(psych)
mlfa<-factanal(covmat=R, factors = 3, rotation="varimax")
mlfa

# Residual Matrix
L=mlfa$loading[, 1:3]
Psi=mlfa$uniquenesses
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

# Factor Loadings Plot
lim<-range(pretty(L))
plot(L[,1], L[,2],main="Plot of Factor Loadings : Varimax",  xlab="f1", ylab="f2",
                       xlim=lim, ylim=lim)
text(L[,1], L[, 2], labels=rownames(L), cex=0.8, col="blue", pos=1)
abline(v=0, h=0)
arrows(0,0, L[,1], L[, 2], col=2, code=2, length=0.1)

# Factor Scores : Regression Method
fml=Z%*%solve(R)%*%L



# PCFA using the principal()
library(psych)
pcfa<-principal(R, nfactors=3, rotate="varimax")
pcfa
round(pcfa$values, 3)
gof=pcfa$values/p*100 # Goodness-of fit
round(gof, 3)

# Residual Matrix
L=pcfa$loading[, 1:3]
Psi=mlfa$uniquenesses
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

# Factor Scores : Regression Method
fpc=Z%*%L%*%solve(t(L)%*%L)
fpc
