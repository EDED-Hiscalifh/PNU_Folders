setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data2.8.2<-read.table("airpollution.txt", header=T)
X=Data2.8.2
rownames<-rownames(X)
p=ncol(X) 
n=nrow(X)
Z<-scale(X, scale=T) # 표준화자료행렬

# Covariance Matrix S(or Correlation Matix R)
R=round(cor(X),3)
R

##### PCFA using the principal( ): 주성분인자분석        #######
library(psych)
pcfa<-principal(Z, nfactors=3, rotate="varimax") # 표준화자료행렬 인자수 3 varimax회전 회귀법인자점수 
pcfa
round(pcfa$values, 3)
gof=pcfa$values/p*100 # Goodness-of fit : 총 기여율
round(gof, 3)

# Residual Matrix : 잔차행렬
L=pcfa$loading[, 1:3] # 인자적재행렬의 추정
round(L, 3)
Psi=pcfa$uniquenesses #특정분산
Rm = R-(L%*%t(L) + diag(Psi)) # 잔차행렬의 추정
round(Rm, 3)

# Plot of PC Factor Loadings : 인자적재그림
par(mfrow=c(2,2))
lim<-range(pretty(L))
plot(L[,1], L[,2],main="(a) PC Factor Loadings : f1 and f2",  xlab="f1", ylab="f2", xlim=lim, ylim=lim)
   text(L[,1], L[, 2], labels=rownames(L), cex=0.8, col="blue", pos=1)
   abline(v=0, h=0)
   arrows(0,0, L[,1], L[, 2], col=2, code=2, length=0.1)
# 생략 : f1과 f2 f1과 f3 

# Factor Scores : Regression Method
fpc=pcfa$scores
round(fpc, 3)

# Plot of Facrtor Scores : PFA : 인자점수그림 
win.graph( )
par(mfrow=c(2,2))
par(pty="s")
lim<-range(pretty(fpc))
plot(fpc[,1], fpc[,2],main=" (a) Factor Scores : f1 and f2",  xlab="f1", ylab="f2", xlim=lim, ylim=lim)
    text(fpc[,1], fpc[,2], labels=rownames(fpc), cex=0.8, col="blue", pos=1)
    abline(v=0, h=0)
# 생략 : f1과 f2 f1과 f3 



##### MLFA using the factanal( ) : 최대우도인자분석####
library(psych)
mlfa<-factanal(Z, factors = 3, rotation="varimax", score="regression") 
mlfa

# Residual Matrix
Lm=mlfa$loading[, 1:3]
round(L, 3)
Psi=mlfa$uniquenesses
Rm = R-(Lm%*%t(Lm) + diag(Psi))
round(Rm, 3)

# ML Factor Loadings Plot
win.graph()
par(mfrow=c(2,2))
lim<-range(pretty(L))
plot(Lm[,1], Lm[,2],main="(a) ML Factor Loadings : f1 and f2",  xlab="f1", ylab="f2", xlim=lim, ylim=lim)
    text(Lm[,1], Lm[, 2], labels=rownames(L), cex=0.8, col="blue", pos=1)
    abline(v=0, h=0)
    arrows(0,0, Lm[,1], Lm[, 2], col=2, code=2, length=0.1)
# 생략 : f1과 f2 f1과 f3 


# Factor Scores : Regression Method
fml=mlfa$scores
round(fml, 3)

# Plot of Facrtor Scores : MLFA
par(mfrow=c(2,2))
par(pty="s")
lim<-range(pretty(fml))
plot(fml[,1], fml[,2],main=" (a) Factor Scores : f1 and f2",  xlab="f1", ylab="f2", xlim=lim, ylim=lim)
   text(fml[,1], fml[,2], labels=rownames(fml), cex=0.8, col="blue", pos=1)
   abline(v=0, h=0)
# 생략 : f1과 f2 f1과 f3 
