library(shapes)
library(biotools)
library(MVN)
library(Hotelling) 
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
mouseT2<-read.in("mouseT2.txt",6,2)
cm<-mouseT2[,,1:30] # 대조그룹 : 30
sm<-mouseT2[,,31:53] # 작은그룹: 23
lm<-mouseT2[,,54:76] # 큰 그룹 : 23

# GPA 
join<-c(1, 6, 2, 3, 4, 5,1)
PAsm<-procGPA(sm)
PAlm<-procGPA(lm)

# Joint Booksteib's Mean Shapes
plot(PAsm$mshape[,1], PAsm$mshape[, 2],xlim=c(-100, 100), ylim=c(-100, 100), xlab="Dim1", ylab="Dim2" )
lines(PAsm$mshape[join,], lty=1)
par(new=TRUE)
plot(PAlm$mshape[,1], PAlm$mshape[, 2],xlim=c(-100, 100), ylim=c(-100, 100), xlab="Dim1", ylab="Dim2" )
lines(PAlm$mshape[join,], lty=3)
legend("bottomright",c("small","laege"), lty=c(1, 3))

# Bookstein Coordinates
booksm<-bookstein2d(sm)
booklm<-bookstein2d(lm)
v1<-t(apply(booksm$bshpv[3:6,,],3,c))
v2<-t(apply(booklm$bshpv[3:6,,],3,c))
p=ncol(v1);

# MVN tests based on the Skewness and Kurtosis Statistics
par(mfrow=c(1,2))
mardiaTest(v1, qqplot = TRUE)
mardiaTest(v2, qqplot = TRUE)

# Data for M-test
n1=dim(sm)[3]
n1
n2=dim(lm)[3]
gp1<-rep(1, n1)
gp2<-rep(2, n2)
small=cbind(v1,gp1)
large=cbind(v2,gp2)
both<-rbind(small,large) 

# Box's M-test for equality of two covariance matrices 
boxM(both[,-9], both[, 9]) 

# Behrens-Fisher Problem : Unequal Covariance Matrices 
S1=cov(v1);S2=cov(v2)
Sp=((n1-1)*S1 + (n2-1)*S2)/(n1+n2-2)
list(S1, S2, Sp)
ms <- matrix(1/n1,1,n1)%*%v1   #작은 그룹 평균
ml <- matrix(1/n2,1,n2)%*%v2   #큰 그룹 평균
e<-(ms-ml)%*%solve((S1/n1)+(S2/n2))
Tsq <- e%*%t(ms-ml);
finv <- (e%*%S1%*%t(e))^2/((Tsq^2)*((n1^3)-(n1^2))) 
       + (e%*%S2%*%t(e))^2/((Tsq^2)*((n2^3)-(n2^2)));
f <- round(1/finv,0)
F0 <- (f-p+1)/(f*p)*Tsq ; 
alpha <- 0.05
Ta <- qf(alpha, p, f-p+1);Ta
pvalue <- 1- pf(F0, p, f-p+1); ue
list(F0, f, Ta, pvalue)


# Hotelling's T^2 test for Bookstein Coordinates
fit=hotelling.test(v1, v2)
fit

