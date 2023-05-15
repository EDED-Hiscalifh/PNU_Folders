# MCRA : Burt Table from Indicator Matrix
Z<-matrix(c(1, 0, 1, 0, 1, 0,
            1, 0, 1, 0, 0, 1,
            1, 0, 1, 0, 0, 1,
            1, 0, 1, 0, 0, 1,
            1, 0, 0, 1, 1, 0,
            1, 0, 0, 1, 1, 0,
            1, 0, 0, 1, 0, 1,
            0, 1, 1, 0, 1, 0,
            0, 1, 1, 0, 0, 1,
            0, 1, 1, 0, 0, 1,
            0, 1, 0, 1, 1, 0,
            0, 1, 0, 1, 0, 1), byrow=T, nrow=12)
colnames(Z)<-c("남", "여", "중장년", "청소년", "키큼", "키작음")
n<-nrow(Z)
q<-ncol(Z)
Z
B <- t(Z) %*% Z 
B 
P <- B / sum(B)
cm <- apply(P, 2, sum)
Dc<-diag(1/sqrt(cm))
eP <- cm %*% t(cm)
Y <- (P - eP) / sqrt(eP)
## Singular Value Decomposition
svd.Y<-svd(Y)
V<-svd.Y$v
Dl<-diag((svd.Y$d)^2)
lam<-(svd.Y$d)^2
fit<-lam/sum(lam)*100
rbind(round(lam, 3),round(fit, 3))
Cb<- Dc%*%V%*%Dl
Cb
par(pty="s")
lim<-range(pretty(Cb))
plot(Cb[, 1:2], xlab="Dim1", ylab="Dim2", xlim=lim, ylim=lim, main="Multiple CA for Burt Matrix")
text(Cb[, 1:2], colnames(Z), pos=3, col=1)
abline(v=0, h=0)

win.graph()
## Spectral Decompsition
dec <- eigen(S)
V<-dec$vectors
delt <- dec$values
expl <- 100*(delt/ sum(delt))
lam <- delt^2
fit <- 100*(lam/ sum(lam))
rbind(roun(lam, 3),round(fit, 1))
Dl<-diag(delt)
Cb<- Dc%*%V%*%Dl
Cb
lim<-range(pretty(Cb))
plot(Cb[, 1:2], xlab="Dim1", ylab="Dim2", main="Multiple CA for Burt Matrix")
text(Cb[, 1:2], colnames(Z), col=2)
abline(v=0, h=0)