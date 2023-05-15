# MVD test and Linear DA for two groups(owner,nonowner)
# of riding mower on two variables(x1.Income x2.Lotsize) 


library(MASS)
library(MVN)

setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
ridingmower<-read.table("ridingmower.txt", header=T)
attach(ridingmower)
ridingmower
# MVN tests based on the Skewness and Kurtosis Statistics
par(mfrow=c(1,2))
owner=ridingmower[1:12, 2:3]
nonowner=ridingmower[13:24, 2:3]
owner
nonowner
result_owner = mardiaTest(owner, qqplot = TRUE)
result_nonowner = mardiaTest(nonowner, qqplot = TRUE)
result_owner
result_nonowner


# Linear DA
LDA=lda(pop~x1.Income + x2.Lotsize, data=ridingmower)
LDA
lcluster=predict(LDA, ridingmower)$class
lct=table(pop, lcluster)
lct
# Total percent correct
mean(pop==lcluster)


# Linear Discriminant Analysis with Jacknifed Prediction
LDACV=lda(pop~x1.Income + x2.Lotsize, data=ridingmower, CV=TRUE)
LDACV
names(LDACV)
cltcv=table(pop, LDACV$class)
cltcv


