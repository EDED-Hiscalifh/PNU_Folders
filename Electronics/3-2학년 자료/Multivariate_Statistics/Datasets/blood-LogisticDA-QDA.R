# Logisitic DA for two groups(건강=1, 비건강=0 )
# on two variables(피보리노겐, 글루블린)
library(DAAG)
library(MVN)
library(biotools)

setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
blood<-read.table("blood.txt", header=T)
attach(blood)
z=blood[, 1:2]
#group=as.numeric(blood[, 3])

# MVN tests based on the Skewness and Kurtosis Ststistics
c1=blood[respon==1,]
c2=blood[respon==0,]
mardiaTest(c1[,1:2])
mardiaTest(c2[,1:2])

#Box's M-Test for Equlaity of Covariance Matrices
boxM(z, respon)

# Quadratic DA: CVM
QDA=qda(respon~., data=blood, prior=c(1,1)/2, CV=TRUE)
# Confusion Table 
confusion=table(blood$respon, QDA$class)
# Expected actual error rate : EAER
EAER=(1-sum(diag(prop.table(confusion))))*100
list(confusion, EAER)

# Quadratic DA : RSM
QDA=qda(respon~., data=blood, prior=c(1,1)/2)
qcluster=predict(QDA, blood)$class
confusion=table(respon, qcluster)
APER=(1-mean(respon==qcluster))*100
list(confusion, APER)

# Logistic DA
blood.glm=glm(respon~., family=binomial, data=blood)
summary(blood.glm)

blood.glm=glm(respon~fibrin, family=binomial, data=blood)
summary(blood.glm)

# Logistic DA based on the fitted Logistic discriminant fuction
logit=-6.8451+0.1109*fibrin
cluster=as.numeric(logit>0)
data.frame(blood, logit, cluster)
confusion=table(respon, cluster) # Confusion Table
APER=(sum(diag(prop.table(confusion))))*100
list(confusion, APER)


# Confusion Table : Cross-Validation method
blood.CV=CVbinary(blood.glm)
CVt=table(respon, round(blood.CV$cv))
EAER=(1-mean(respon==round(blood.CV$cv)))*100
list(CVt, EAER)

# Confusion Table : Resubstitution nethod 
Rt=table(respon, round(blood.CV$internal))
APER=(1-mean(respon==round(blood.CV$internal)))*100
list(Rt, APER)
