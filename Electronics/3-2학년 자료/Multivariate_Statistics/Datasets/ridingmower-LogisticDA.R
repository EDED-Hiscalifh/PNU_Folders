library(DAAG)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
ridingmower<-read.table("ridingmower.txt", header=T)
attach(ridingmower)
z=ridingmower[, 2:3]
group=as.numeric(ridingmower$pop<=1)
ridingmower=data.frame(z, pop=group)

rdm.glm=glm(pop~x1.Income + x2.Lotsize, family=binomial, data=ridingmower)
summary(rdm.glm)

# Logistic DA based on the fitted Logistic discriminant fuction
logit=-25.9382+0.1109*x1.Income+0.9638*x2.Lotsize
cluster=as.numeric(logit>0)
data.frame(ridingmower, logit, cluster)
confusion=table(pop, cluster) # Confusion Table
APER=(sum(diag(prop.table(confusion))))*100
list(confusion, APER)

# Confusion Table : Cross-Validation method
rdm.CV=CVbinary(rdm.glm)
names(rdm.CV)
CVt=table(pop, round(rdm.CV$cv))
EAER=(sum(diag(prop.table(CVt))))*100
list(CVt,EAER)

# Confusion Table : Resubstitution method 
Rt=table(pop, round(rdm.CV$internal))
APER=(sum(diag(prop.table(Rt))))*100
list(Rt, APER)
