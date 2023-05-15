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