
R=matrix(c(1.00, 0.83, 0.78, 0.70, 0.66, 0.63,
           0.83, 1.00, 0.84, 0.67, 0.65, 0.57,
           0.78, 0.84, 1.00, 0.64, 0.54, 0.51,
           0.70, 0.67, 0.64, 1.00, 0.45, 0.51,
           0.66, 0.65, 0.54, 0.45, 1.00, 1.00,
           0.63, 0.57, 0.51, 0.51, 0.40, 1.00), byrow=T, nrow=6)

rowname=c("고전","불어","영어","수학","음감", "음악")
#R1=R[1:3,1:3]
#R1
#R=R1
eigen.R=eigen(R)
eigen.R
round(eigen.R$values, 3) # Eigenvalues
L=round(eigen.R$vectors, 3) # Eigenvaectors
L
LL=t(L)%*%L
Psi = R-LL
Psi
gof=eigen.R$values/sum(eigen.R$values)*100 # Goodness-of-fit
round(gof, 2)

