# Box’s test for the hypothesis test of the equality of covariance matrices
# in Riding-Mowers data.
library(MASS)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
ridingmower<-read.table("ridingmower.txt", header=T)
attach(ridingmower)

# Scatter Plot
plot(ridingmower[, 2:3], pch=unclass(pop))
# Density function of Variables
par(mfrow=c(1,2))
ldahist(data= x1.Income, g=pop, width=30, type="density")
ldahist(data= x2.Lotsize, g=pop, width=10, type="density")

cov.Mtest=function(x,ina,a=0.05){
## x is the data set
## ina is a numeric vector indicating the groups of the data set
## a is the significance level, set to 0.05 by default
x=as.matrix(x)
p=ncol(x) ## dimension of the data set
n=nrow(x) ## total sample size
k=max(ina) ## number of groups
nu=rep(0,k) ## the sample size of each group will be stored here later
pame=rep(0,k) ## the determinant of each covariance will be stored here
## the next "for" function calculates the covariance matrix of each group
nu=as.vector(table(ina))
mat=mat1=array(dim=c(p,p,k))
for (i in 1:k) {
mat[,,i]=cov(x[ina==i,])
pame[i]=det(mat[,,i]) ## the detemirnant of each covariance matrix
mat1[,,i]=(nu[i]-1)*cov(x[ina==i,]) }
## the next 2 lines calculate the pooled covariance matrix
Sp=apply(mat1,1:2,sum)
Sp=Sp/(n-k)
for (i in 1:k)
pamela=det(Sp) ## determinant of the pooled covariance matrix
test1=sum((nu-1)*log(pamela/pame))
gama1=(2*(p^2)+3*p-1)/(6*(p+1)*(k-1))
gama2=(sum(1/(nu-1))-1/(n-k))
gama=1-gama1*gama2
test=gama*test1 ## this is the M (test statistic)
df=0.5*p*(p+1)*(k-1) ## degrees of freedom of the chi-square distribution
pvalue=1-pchisq(test,df) ## p-value of the test statistic
crit=qchisq(1-a,df) ## critical value of the chi-square distribution
list(M.test=test,degrees=df,critical=crit,p.value=pvalue) }

ina=as.numeric(ridingmower[, 1])
x=ridingmower[, 2:3]
cov.Mtest(x, ina)
