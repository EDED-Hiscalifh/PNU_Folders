#CCA Steps for First and Second Sons' Head
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")

#[Step 1] Data Matrix 
Data4.2.1<-read.table("head.txt", header=T)
Data4.2.1
n=nrow(Data4.2.1)
# Sets of Variables : X, Y
X=Data4.2.1[,1:2] #First sons
X=scale(X, scale=F)
X
Y=Data4.2.1[,3:4] 
Y=scale(Y, scale=F) # Second sons
Y

#[Step 2] Covariance Matrix S(or Correlation Matix R)
S=cov(Data4.2.1)
round(S, 3)
Sxx=cov(X)
Syy=cov(Y)
Sxy=t(X)%*%Y/(n-1)
Syx=t(Sxy)

# Sx ^ Sy
Exx <- eigen(Sxx)
Eyy <- eigen(Syy)
Sx <- Exx$vectors %*% diag(sqrt(Exx$values)) %*% t(Exx$vectors)
Sy <- Eyy$vectors %*% diag(sqrt(Eyy$values)) %*% t(Eyy$vectors)

round(Sxx, 3)
round(Syy, 3)
round(Sxy, 3)

#[Step 3] Spectral Decomposition : M=PDP
M=solve(Sx)%*%Sxy%*%solve(Syy)%*%Syx%*%solve(Sx)
round(M, 3)
eigen.M=eigen(M)
eig=eigen.M$values
P=eigen.M$vectors

per <- eig/sum(eig)*100
gof <- sum(per[1:2])
per
gof
round(eig, 3) # Eigenvalues
round(sqrt(eig), 3) #Canonical Correlation

U=round(solve(Sx)%*%P, 3) # Eigenvectors
rownames(U)<-colnames(X)
U
V=solve(Syy)%*%Syx%*%U%*%diag(1/sqrt(eig))
V

#[Step 4] Canonical Varables Scores
Zx=X%*%U[, 1:2]
Zx
Zy=Y%*%V[, 1:2]
Zy

# Plot of Canonical Variables and Scores  
par(mfrow=c(1,2))
par(pty="s")
lim1 <- range(pretty(Zx))
lim2 <- range(pretty(Zy))
biplot(Zx, U, xlab="1st Dimemsion",ylab="2nd Dimension", main=" (a) First Sons", xlim=lim1,ylim=lim1,cex=0.8,pch=16)
abline(v=0,h=0)
biplot(Zy, V, xlab="1st Dimemsion",ylab="2nd Dimension", main=" (b) Second Sons", xlim=lim2,ylim=lim2,cex=0.8,pch=16)
abline(v=0,h=0)