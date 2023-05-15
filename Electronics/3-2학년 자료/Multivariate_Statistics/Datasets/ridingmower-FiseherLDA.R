# Fisher Linear DA for two groups(owner,nonowner)
# of riding mower on two variables(x1.Income x2.Lotsize) 
library(MASS)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
ridingmower<-read.table("ridingmower.txt", header=T)
attach(ridingmower)
ridingmower
z=ridingmower[, 2:3]
group=as.numeric(ridingmower[, 1])

FLDA=function(z,group) {
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
W=apply(S,1:2,sum) ## Within sum of squares
B=apply(B1,1:2,sum) ## Between sum of squares
invW=solve(W)
M=invW%*%B
eigen.M=eigen(M)
V=eigen.M$vectors
v1=V[,1]
z=as.matrix(z)
for (j in 1:n) {
w=matrix(z[j,],d,1)
like=rep(0,k)
for (m in 1:k) {
like[m]=abs(v1%*%w-v1%*%mat[,m]) }
pred[j]=which.min(like) } ## The predicted group
cluster=table(group, pred)
APER=1-mean(group==pred)
list(xbar=xbar, B=B, W=W, invW=invW, M=M, eigen=eigen(M),V=V,
table=cluster, APER=APER) }

FLDA(z, group)

# Discriminant Scores
v1=FLDA(z, group)$V[,1]
z=as.matrix(z)
y=z%*%v1
y
mean(y)
plot(z[, 1], y, xlab="X1.Income", ylab="제1판별변수" )
text(z[, 1], y, labels=group, cex=0.8, col=2, pos=1)