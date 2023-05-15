# Linear and Quadratic DA for two groups(setosa, vesicolor)
#   on two variables(꽃받침길이,꽃받침폭) 
library(MASS)
library(MVN)

setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
setosa_versi<-read.table("setosaversicolor.txt", header=T)
attach(setosa_versi)



# MVN tests based on the Skewness and Kurtosis Statistics
par(mfrow=c(1,2))
setosa=setosa_versi[1:50, 1:2]
versicolor=setosa_versi[51:100, 1:2]
setosa
versicolor
result_setosa = mardiaTest(setosa, qqplot = TRUE)
result_versicolor = mardiaTest(versicolor, qqplot = TRUE)
result_setosa
result_versicolor


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
