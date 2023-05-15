library(biotools)

setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
fisherexample<-read.table("fisherexample.txt", header=T)
attach(fisherexample)
fisherexample
z=fisherexample[, 1:2]
z
group=as.numeric(fisherexample[, 3])

#Box's M-Test for Equlaity of Covariance Matrices
boxM(z, group)

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

# Discriminant Scores and Scatter Plot
V=FLDA(z, group)$V
z=as.matrix(z)
y=z%*%V
centroid=aggregate(y, by=list(group), mean)
list(y, centroid) 

plot(y[, 1], y[,2], xlab="제1판별변수", ylab="제2판별변수" )
text(y[, 1], y[,2], labels=group, cex=0.8, col=2, pos=1)