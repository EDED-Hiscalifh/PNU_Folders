#CCA Steps for 5 Subjects

#[Step 1] Data Matrix 
library("MVT")
data(examScor)

# Sets of Variables : X, Y
X=examScor[,1:2] # Closed books
X=scale(X, scale=T)
Y=examScor[,3:5] # Opened books
Y=scale(Y, scale=T)

#[Step 2] Covariance Matrix S(or Correlation Matix R)
R=round(cor(examScor),3)
R

Rxx=R[1:2, 1:2]
Ryy=R[3:5, 3:5]
Rxy=R[1:2, 3:5]
Ryx=t(Rxy)
Rxx
Ryy
Rxy

# Rx ^ Ry
Exx <- eigen(Rxx)
Eyy <- eigen(Ryy)
Rx <- Exx$vectors %*% diag(sqrt(Exx$values)) %*% t(Exx$vectors)
Ry <- Eyy$vectors %*% diag(sqrt(Eyy$values)) %*% t(Eyy$vectors)


#[Step 3] Spectral Decomposition : M=PDP
M=solve(Rx)%*%Rxy%*%solve(Ryy)%*%Ryx%*%solve(Rx)
round(M, 3)
eigen.M=eigen(M)
eig=eigen.M$values
P=eigen.M$vectors

round(eig, 3) # Eigenvalues
round(sqrt(eig), 3) #Canonical Correlation
U=round(solve(Rx)%*%P, 3) # Eigenvectors
rownames(U)<-colnames(X)
U
V=solve(Ryy)%*%Ryx%*%U%*%diag(1/sqrt(eig))
V

#[Step 4] Canonical Varables Scores
Zx=X%*%U
Zx
Zy=Y%*%V
Zy

# Plot of Canonical Variables and Scores  
par(mfrow=c(1,2))
par(pty="s")
lim1 <- range(pretty(Zx))
lim2 <- range(pretty(Zy))
biplot(Zx, U, xlab="1st Dimemsion",ylab="2nd Dimension", main=" (a) Closed-Books", xlim=lim1,ylim=lim1,cex=0.8,pch=16)
abline(v=0,h=0)
biplot(Zy, V, xlab="1st Dimemsion",ylab="2nd Dimension", main=" (b) Open-Books", xlim=lim2,ylim=lim2,cex=0.8,pch=16)
abline(v=0,h=0)


