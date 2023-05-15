library(psych)
data(Harman74.cor)
R=Harman74.cor

mlfa<-factanal(covmat=R, factors = 2)
mlfa
