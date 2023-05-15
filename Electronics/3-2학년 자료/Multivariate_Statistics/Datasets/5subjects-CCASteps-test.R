#CCA Test for 5 Subjects
#[Step 1] Data Matrix 
library("MVT")
data(examScor)

# Sets of Variables : X, Y
X=examScor[,1:2] # Closed books
X=scale(X, scale=T)
Y=examScor[,3:5] # Opened books
Y=scale(Y, scale=T)
n=nrow(X)
p=ncol(X)
q=ncol(Y)

#[Step 2] Covariance Matrix S(or Correlation Matix R)
R=round(cor(examScor),3)
R

Rxx=R[1:2, 1:2]
Ryy=R[3:5, 3:5]
Rxy=R[1:2, 3:5]
Ryx=t(Rxy)

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
round(eig, 3) # Eigenvalues
round(sqrt(eig), 3) #Canonical Correlation

#[Step 4] Testing Significant Canonical Correlations
ev <- (1 - eigen.M$values)
s <- min(p, q)
s
w=n - 3/2 - (p + q)/2
Lambda <- rev(cumprod(rev(ev)))
# initialize
d1 <- d2 <- f <- vector("numeric", s)
for (i in 1:s) {
    t <- sqrt((p^2 * q^2 - 4)/(p^2 + q^2 - 5))
    ti <- 1/t
    d1[i] <- p * q
    d2[i] <-w * t - p * q/2 + 1
    r <- (1 - Lambda[i]^ti)/Lambda[i]^ti
    f[i] <- r * d2[i]/d1[i]
    p <- p - 1
    q <- q - 1
}

p_value <- pf(f, d1, d2, lower.tail = FALSE)
(dmat <- cbind(WilksLambda = Lambda, F = f, df1 = d1, df2 = d2, p_value = p_value))


