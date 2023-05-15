#CCA Steps for Korean Tennis Players
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")

#[Step 1] Data Matrix 
Data3.9.1<-read.table("tennis.txt", header=T)
Data3.9.1
n=nrow(Data3.9.1)
# Sets of Variables : X, Y
X=Data3.9.1[,1:8] #체격변수자료
X=scale(X, scale=T)
Y=Data3.9.1[,9:16] # 체략변수자료
Y=scale(Y, scale=T)
Y

#[Step 2] Covariance Matrix S(or Correlation Matix R)
Rxx=cor(X)
Ryy=cor(Y)
Rxy=t(X)%*%Y/(n-1)
Ryx=t(Rxy)
Rxx
Ryy
Rxy

#[Step 3] Spectral Decomposition : M
M=solve(Rxx)%*%Rxy%*%solve(Ryy)%*%Ryx
eigen.M=eigen(M)
eigen.M
eig=eigen.M$values
per <- eig/sum(eig)*100
gof <- sum(per[1:2])
per
gof
round(eig, 3) # Eigenvalues
round(sqrt(eig), 3) #Canonical Correlation

U=round(eigen.M$vectors, 3) # Eigenvectors
rownames(U)<-colnames(X)
U
V=solve(Ryy)%*%Ryx%*%U%*%diag(1/sqrt(eigen.M$values))
V

#[Step 4] Canonical Varables Scores
Zx=X%*%U[, 1:2]
Zx
Zy=Y%*%V[, 1:2]
Zy

# Plot of Canonical Variables and Scores  
par(mfrow=c(1,2))
par(pty="s")
biplot(Zx, U, xlab="1st Dimemsion",ylab="2nd Dimension", main=" (a) 체격변수", xlim=c(-0.25, 0.2),ylim=c(-0.07, 0.1),cex=0.8,pch=16)
abline(v=0,h=0)
biplot(Zy, V, xlab="1st Dimemsion",ylab="2nd Dimension", main=" (b) 체력변수", xlim=c(-0.15, 0.15),ylim=c(-0.07, 0.07),cex=0.8,pch=16)
abline(v=0,h=0)


