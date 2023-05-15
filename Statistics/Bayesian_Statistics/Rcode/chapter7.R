library(MASS) ; source("hdr2d.R")

#### Simulate multivariate normal vector
rmvnorm<-
function(n,mu,Sigma) {
  p<-length(mu)
  res<-matrix(0,nrow=n,ncol=p)
  if( n>0 & p>0 ) {
  E<-matrix(rnorm(n*p),n,p)
  res<-t(  t(E%*%chol(Sigma)) +c(mu))
                   }
  res
                       }

#### Simulate inverse-Wishart matrix
rinvwish<-function(n,nu0,iS0) 
{
  sL0 <- chol(iS0) 
  S<-array( dim=c( dim(L0),n ) )
  for(i in 1:n) 
  {
     Z <- matrix(rnorm(nu0 * dim(L0)[1]), nu0, dim(iS0)[1]) %*% sL0  
     S[,,i]<- solve(t(Z)%*%Z)
  }     
  S[,,1:n]
}



#### Log density of the multivariate normal distribution
ldmvnorm<-function(y,mu,Sig){  # log mvn density
       c(  -(length(mu)/2)*log(2*pi) -.5*log(det(Sig)) -.5*
            t(y-mu)%*%solve(Sig)%*%(y-mu)   )  
                             }


#### Simulate from the Wishart distribution
rwish<-function(n,nu0,S0)
{
  sS0 <- chol(S0)
  S<-array( dim=c( dim(S0),n ) )
  for(i in 1:n)
  {
     Z <- matrix(rnorm(nu0 * dim(S0)[1]), nu0, dim(S0)[1]) %*% sS0
     S[,,i]<- t(Z)%*%Z
  }
  S[,,1:n]
}



#### Figure 


mu<-c(50,50)
sig<-c(8,12)
ng<-50
par(mfrow=c(1,3),mar=c(2.75,2.75,.5,.5),mgp=c(1.70,.70,0))
y<-seq(20,80,length=ng)
set.seed(1)

for( rho in c(-.5,0,.5)) {

Sigma<- matrix( c(1,rho,rho,1),2,2) * outer(sig,sig)

Y<-rmvnorm(30,mu,Sigma)

LDY<-matrix(nrow=ng,ncol=ng)
for(i in 1:ng){
for(j in 1:ng){ LDY[i,j]<-ldmvnorm( c(y[i],y[j]),mu,Sigma) }}

plot(range(y),range(y),type="n", 
  xlab=expression(italic(y[1])),ylab=expression(italic(y[2])) )
filledcontour(y,y,exp(LDY))
points(Y,pch="x")
                           }




#### Reading comprehension
load("reading.RData")
Y<-reading

mu0<-c(50,50)
L0<-matrix( c(625,312.5,312.5,625),nrow=2,ncol=2)

nu0<-4
S0<-matrix( c(625,312.5,312.5,625),nrow=2,ncol=2)

n<-dim(Y)[1] ; ybar<-apply(Y,2,mean)
Sigma<-cov(Y) ; THETA<-SIGMA<-NULL
YS<-NULL
set.seed(1)

for(s in 1:5000) 
{
 
  ###update theta
  Ln<-solve( solve(L0) + n*solve(Sigma) )
  mun<-Ln%*%( solve(L0)%*%mu0 + n*solve(Sigma)%*%ybar )
  theta<-rmvnorm(1,mun,Ln)  
  ### 
   
  ###update Sigma
  Sn<- S0 + ( t(Y)-c(theta) )%*%t( t(Y)-c(theta) ) 
#  Sigma<-rinvwish(1,nu0+n,solve(Sn))
  Sigma<-solve( rwish(1, nu0+n, solve(Sn)) )
  ###

  ###
  YS<-rbind(YS,rmvnorm(1,theta,Sigma)) 
  ###

  ### save results 
  THETA<-rbind(THETA,theta) ; SIGMA<-rbind(SIGMA,c(Sigma))
  ###
  cat(s,round(theta,2),round(c(Sigma),2),"\n")
}

quantile(  SIGMA[,2]/sqrt(SIGMA[,1]*SIGMA[,4]), prob=c(.025,.5,.975) )
quantile(   THETA[,2]-THETA[,1], prob=c(.025,.5,.975) )
mean( THETA[,2]-THETA[,1])
mean( THETA[,2]>THETA[,1]) 
mean(YS[,2]>YS[,1])


#### Figure
library(ash)

par(mfrow=c(1,2),mgp=c(1.75,.75,0),mar=c(3,3,1,1))

plot.hdr2d(THETA,xlab=expression(theta[1]),ylab=expression(theta[2]) )
abline(0,1)

plot.hdr2d(YS,xlab=expression(italic(y[1])),ylab=expression(italic(y[2])), 
     xlim=c(0,100),ylim=c(0,100) )
points(Y[,1],Y[,2],pch=16,cex=.7)
abline(0,1)




#### Pima data 
data(Pima.tr)
Y0<-Pima.tr[,2:5]
Y<-Y0
n<-dim(Y)[1]
p<-dim(Y)[2]

set.seed(1)
O<-matrix(rbinom(n*p,1,.9),n,p)
Y[O==0]<-NA


pdf("fig7_3.pdf",family="Times", height=6,width=6)
par(mar=c(1,1,.5,.5)*1.75,mfrow=c(p,p),mgp=c(1.75,.75,0))
for(j1 in 1:p) {
for(j2 in 1:p) {
 if(j1==j2){hist(Y[,j1],main="");mtext(colnames(Y)[j1],side=3,line=-.1,cex=.7)}
  if(j1!=j2) { plot(Y[,j1],Y[,j2],xlab="",ylab="",pch=16,cex=.7)} 
                }}



####

## prior parameters
p<-dim(Y)[2]
mu0<-c(120,64,26,26)
sd0<-(mu0/2)
L0<-matrix(.1,p,p) ; diag(L0)<-1 ; L0<-L0*outer(sd0,sd0)
nu0<-p+2 ; S0<-L0
###

### starting values
Sigma<-S0
Y.full<-Y
for(j in 1:p)
{
  Y.full[is.na(Y.full[,j]),j]<-mean(Y.full[,j],na.rm=TRUE)
}
###


### Gibbs sampler
THETA<-SIGMA<-Y.MISS<-NULL
set.seed(1)

for(s in 1:1000)
{

  ###update theta
  ybar<-apply(Y.full,2,mean)
  Ln<-solve( solve(L0) + n*solve(Sigma) )
  mun<-Ln%*%( solve(L0)%*%mu0 + n*solve(Sigma)%*%ybar )
  theta<-rmvnorm(1,mun,Ln)
  ###
  
  ###update Sigma
  Sn<- S0 + ( t(Y.full)-c(theta) )%*%t( t(Y.full)-c(theta) )
#  Sigma<-rinvwish(1,nu0+n,solve(Sn))
  Sigma<-solve( rwish(1, nu0+n, solve(Sn)) )
  ###
  
  ###update missing data
  for(i in 1:n)
  { 
    b <- ( O[i,]==0 )
    a <- ( O[i,]==1 )
    iSa<- solve(Sigma[a,a])
    beta.j <- Sigma[b,a]%*%iSa
    s2.j   <- Sigma[b,b] - Sigma[b,a]%*%iSa%*%Sigma[a,b]
    theta.j<- theta[b] + beta.j%*%(t(Y.full[i,a])-theta[a])
    Y.full[i,b] <- rmvnorm(1,theta.j,s2.j )
  }
  
  ### save results
  THETA<-rbind(THETA,theta) ; SIGMA<-rbind(SIGMA,c(Sigma))
  Y.MISS<-rbind(Y.MISS, Y.full[O==0] )
  ###

  cat(s,theta,"\n")
}

#### Posterior mean and correlation matrix
apply(THETA,2,mean)

COR <- array( dim=c(p,p,1000) )
for(s in 1:1000)
{
  Sig<-matrix( SIGMA[s,] ,nrow=p,ncol=p)
  COR[,,s] <- Sig/sqrt( outer( diag(Sig),diag(Sig) ) )
}

apply(COR,c(1,2),mean)



