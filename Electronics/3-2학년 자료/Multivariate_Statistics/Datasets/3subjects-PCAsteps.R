setwd("c:/R과 함께하는 다변량자료분석/R_code_data")

#[Step 1] Data Matrix X
Data1.1.1<-read.table("3subjects.txt", header=T)
X<-Data1.1.1
rownames(X)

#[Step 2] Covariance Matrix S(or Correlation Matix R)
S=round(cov(X),3)
S

#[Step 3] Spectrla Decompositoin 
eigen.S=eigen(S)
round(eigen.S$values, 3) # Eigenvalus
V=round(eigen.S$vectors, 3) # Eigenvaectors
V
#[Step 4] Choice of Eigenvalues and Eigenvectors
gof=eigen.S$values/sum(eigen.S$values)*100 # Goodness-of fit
round(gof, 2)

#[Step 5] PCs : liner combination of original variables 
V2=V[,1:2]
V2
#[Step 6] PCS, PCs Scores and New Data Matrix P
Y=scale(X, scale=F) # Centred Data Matrix
P=Y%*%V2            # PCs Scores
P

#[Step 7] Plot of PCs Scores
par(pty="s")
lim<-range(pretty(P))
plot(P[,1], P[, 2], main="Plot of PCs Scores", xlim=lim, ylim=lim, 
           xlab="1st PC", ylab="2nd PC")
text(P[,1]+2, P[, 2], rownames(X), cex=0.8, col="blue", pos=3)
abline(v=0, h=0)
