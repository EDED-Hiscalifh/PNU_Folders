library(HDclassif)
library("MVN")
library(biotools)

#[Step 1] Wine data : clusters 1, 2, 3
data(wine)
attach(wine)
pairs(wine[2:14], pch=21, bg=c("red", "green", "blue")[unclass(wine$class)])

#[Step 2] MVN tests based on the Skewness and Kurtosis Ststistics
c1 = wine[1:59, 2:14]
c2 = wine[60:130,2:14] 
c3 = wine[131:178,2:14]
result_c1 = mardiaTest(c1, qqplot = TRUE)
result_c2 = mardiaTest(c2, qqplot = TRUE)
result_c3 = mardiaTest(c3, qqplot = TRUE)
list(result_c1,result_c2,result_c3) 

#[Step 3] Box's M-Test for Equlaity of Covariance Matrices
boxM(wine[, -1], wine[, 1])
S1=cov(c1);S2=cov(c2);S3=cov(c3)
Sp=(58*S1+70*S2+47*S3)/(178-3)
list(S1, S2, S3, Sp)

#[Step 4] Quadratic DA with 3 groups applying 
# resubstitution prediction and equal prior probabilities.
QDA=qda(class~., data=wine, prior=c(1,1,1)/3)
qcluster=predict(QDA, wine)$class
table(class, qcluster)

# Total percent correct
mean(class==qcluster)


