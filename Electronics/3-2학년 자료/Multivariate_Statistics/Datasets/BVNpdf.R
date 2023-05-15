BVNpdf <- function(mu1,mu2,sig1,sig2,rho) {
par(mfrow=c(1,2))
s12 = sig1*sig2*rho
s11 = sig1^2
s22 = sig2^2
Sig <- matrix(c(s11,s12,s12,s22),ncol=2,nrow=2,byrow=T)
Sinv <- solve(Sig)
x1 <- seq(mu1 - 3.5*sig1,mu1+3.5*sig1,len=50)
fx1 <- seq(-3.5,3.5,len=50)
x2 <- seq(mu2 - 3.5*sig2,mu2+3.5*sig2,len=50)
fx2 <- seq(-3.5,3.5,len=50)
f <- function(x1,x2) {    
                cons <- ((2*pi)*det(Sig)^.5)^{-1}
                cons*exp(-(.5*(1 - rho^2)^{-1})*(x1^2+x2^2-2*rho*x1*x2))
        }
f <- outer(fx1,fx2,f)
persp(x1,x2,f,theta = 30, expand=.50)
title(main="Bivariate Normal pdf")
contour(x1,x2,f,lty="solid",drawlabels=F)
title(main="Contour Plot of BVN pdf")
BVNpdf(mu1,mu2,sig1,sig2,rho)
par(mfrow=c(1,1))
}

BVNpdf(0,0,1,1,0)
BVNpdf(0,0,1,1,0.8)
BVNpdf(0,0,1,1,-0.8)


