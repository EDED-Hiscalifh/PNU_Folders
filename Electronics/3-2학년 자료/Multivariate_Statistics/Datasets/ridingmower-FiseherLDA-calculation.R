# MVD test and Linear DA for two groups(owner,nonowner)
# of riding mower on two variables(x1.Income x2.Lotsize) 
library(MASS)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
ridingmower<-read.table("ridingmower.txt", header=T)
attach(ridingmower)

setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
setosa_versi<-read.table("setosaversicolor.txt", header=T)
attach(setosa_versi)

setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
fisherexample<-read.table("fisherexample.txt", header=T)
attach(fisherexample)
fisherexample
z=fisherexample[, 1:2]
group=as.numeric(fisherexample[, 3])
z
group

z=ridingmower[, 2:3]
group=as.numeric(ridingmower[, 1])

z=setosa_versi[, 1:2]
group=as.numeric(setosa_versi[, 3])
group

## z contains the data
## group denotes the groups
k=max(group) ; n=nrow(z)
d=ncol(z) ; pred=rep(0,n)
x=z ; ina=group
xbar=colMeans(x)
S=array(dim=c(ncol(x),ncol(x),k))
B1=array(dim=c(ncol(x),ncol(x),k))
mat=matrix(rep(0,d*k),nrow=d,ncol=k)
for (i in 1:k) {
S[,,i]=(nrow(x[ina==i,])-1)*cov(x[ina==i,])
B1[,,i]=( (colMeans(x[ina==i,])-xbar)%*%t(colMeans(x[ina==i,])-xbar) )
mat[,i]=colMeans(x[ina==i,]) }
W=apply(S,1:2,sum) ## The within sum of squares
B=apply(B1,1:2,sum) ## The between sum of squares
solve(W)
M=solve(W)%*%B
eigen.M=eigen(M)
V=eigen.M$vectors
v1=V[,1]
as.vector(eigen(M)$vectors[,1]) ## Fisher’s discriminant function
z=as.matrix(z)
for (j in 1:n) {
w=matrix(z[j,],d,1)
like=rep(0,k)
for (m in 1:k) {
like[m]=abs(v1%*%w-v1%*%mat[,m]) }
pred[j]=which.min(like) } ## The predicted group
cluster=table(group, pred)
APER=1-mean(group==pred)
list(xbar=xbar, B=B, W=W, invW=solve(W), M=M, eigen=eigen(M),V=V,
table=cluster, APER=APER) 

# Discriminant Scores
y=z%*%v1
mean(y)
plot(z[, 1], y, xlab="X1.Income", ylab="제1판별변수" )
text(z[, 1], y, labels=group, cex=0.8, col=2, pos=1)

