# Linear and Quadratic DA for two groups(setosa, vesicolor)
#   on two variables(꽃받침길이,꽃받침폭) 

setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
setosa_versi<-read.table("setosaversicolor.txt", header=T)
attach(setosa_versi)
plot(setosa_versi[, 1:2], pch=unclass(종류), col=1:2)
library(MASS)
# Density function of Variables
ldahist(data= 꽃받침길이, g=종류, type="density")
ldahist(data= 꽃받침폭, g=종류, type="density")

# Linear DA
LDA=lda(종류~꽃받침길이+꽃받침폭, data=setosa_versi)
LDA
lcluster=predict(LDA, setosa_versi)$class
lct=table(종류, lcluster)
lct
# Total percent correct
mean(종류==lcluster)

# Linear Discriminant Analysis with Jacknifed Prediction
LDACV=lda(종류~꽃받침길이+꽃받침폭, data=setosa_versi, CV=TRUE)
LDACV
names(LDACV)
cltcv=table(종류, LDACV$class)
cltcv


# Quadratic DA with 3 groups applying 
 # resubstitution prediction and equal prior probabilities.
QDA=qda(종류~꽃받침길이+꽃받침폭, data=setosa_versi, prior=c(1,1)/2)
QDA
qcluster=predict(QDA, setosa_versi)$class
qct=table(종류, qcluster)
qct
# Total percent correct
mean(종류==qcluster)


