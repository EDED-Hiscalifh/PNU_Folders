library(shapes)
library(biotools)
library(MVN)
library(Hotelling) 
# female and male Gorillas (Dryden and Mardia, 1998)
data(gorf.dat)
data(gorm.dat)
n1<-dim(gorf.dat)[3]
n2<-dim(gorm.dat)[3]

gorft<-array(0,dim=c(8,2, n1))
for(i in 1:n1){
T<-matrix(c(cos(90),sin(90),
           -sin(90), cos(90)), byrow=T, nrow=2)
gorft[,,i]<-gorf.dat[,,i]%*%T
}
gormt<-array(0,dim=c(8,2, n2))
for(i in 1:n2){
T<-matrix(c(cos(90), sin(90),
           -sin(90), cos(90)), byrow=T, nrow=2)
gormt[,,i]<-gorm.dat[,,i]%*%T
}

join<-c(1,5,4,3,2,8,7,6,1)
plotshapes(gorft, gormt, symbol=1,joinline=join)

bookgorf<-bookstein2d(gorft)
bookgorm<-bookstein2d(gormt)

v1<-t(apply(bookgorf$bshpv[3:8,,],3,c))
v2<-t(apply(bookgorm$bshpv[3:8,,],3,c))
S1=cov(v1);S2=cov(v2)
Sp=((n1-1)*S1 + (n2-1)*S2)/(n1+n2-2)
list(S1, S2, Sp)

# Data for M-test
gpf<-rep(1, n1)
gpm<-rep(2, n2)
female=cbind(v1,gpf)
male=cbind(v2,gpm)
both<-rbind(female,male) 

# MVN tests based on the Skewness and Kurtosis Statistics
par(mfrow=c(1,2))
mardiaTest(v1, qqplot = TRUE)
mardiaTest(v2, qqplot = TRUE)

# Box's M-test for equality of two covariance matrices 
boxM(both[,-13], both[, 13]) 

# Hotelling's T^2 test for Bookstein Coordinates
fit=hotelling.test(v1, v2)
fit



