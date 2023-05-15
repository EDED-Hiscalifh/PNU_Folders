#CCA for 5 Subjects

# Data Matrix 
library("MVT")
data(examScor)

# Sets of Variables : X, Y
X=examScor[,1:2] # Closed books
X=scale(X, scale=T)
Y=examScor[,3:5] # Opened books
Y=scale(Y, scale=T)
Y

# CCA using the cc( )
library(CCA)
cca=cc(X, Y)
cca
plt.cc(cca, type="b", var.label=T)


# CCA using the cancor( )
cca<-cancor(X, Y)
cca
# Rows and Columns Coordinates : Rx Cx Ry Cy
Rx <- X%*%cca$xcoef[,1:2]
Cx <- cca$xcoef[,1:2]
Ry <- Y%*%cca$ycoef[,1:2]
Cy <- cca$ycoef[,1:2]
rownames(Cx) <- colnames(X) 
rownames(Cy) <- colnames(Y)

par(mfrow=c(1,2))
par(pty="s")
biplot(Rx, Cx, xlab="1st Dimension", ylab="2nd Dimension",
     main=" (a) Closed-Books",  xlim=c(-0.5, 0.5),ylim=c(-0.3, 0.3),cex=0.8,pch=16)
abline(v=0,h=0)
biplot(Ry, Cy, xlab="1st Dimension", ylab="2nd Dimension", 
  main=" (b) Open-Books",  xlim=c(-0.5, 0.5),ylim=c(-0.3, 0.3),cex=0.8,pch=16)
abline(v=0,h=0)

