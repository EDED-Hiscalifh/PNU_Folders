# MLFA Steps for KLPGA

# Data Matrix X
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X=Data1.3.2
rownames<-rownames(X)
p=ncol(X) 

# Covariance Matrix S(or Correlation Matix R)
R=round(cor(X),3)
R

# ML Estimation using the factanal( )
library(psych)
mlfa<-factanal(covmat=R, factors = 2,  rotation="varimax" ) # rotation="none" 
mlfa

# Residual Matrix
L=mlfa$loading[, 1:2]
L
Psi=mlfa$uniquenesses
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

# Factor Loadings Plot
lim<-range(pretty(L))
plot(L[,1], L[,2],main="Plot of Factor Loadings : Varimax ",  xlab="f1", ylab="f2",
                       xlim=lim, ylim=lim)
text(L[,1], L[, 2], labels=rownames(L), cex=0.8, col="blue", pos=1)
abline(v=0, h=0)
arrows(0,0, L[,1], L[, 2], col=2, code=2, length=0.1)




