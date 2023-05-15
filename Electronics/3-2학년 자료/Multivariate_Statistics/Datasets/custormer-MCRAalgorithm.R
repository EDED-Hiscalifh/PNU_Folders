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
rownames(Cb)<-colnames(Z)
Cb2=Cb[, 1:2]
Cb2
par(pty="s")
lim<-range(pretty(Cb))
plot(Cb2, xlab="Dim1", ylab="Dim2", xlim=lim, ylim=lim, main="MCRA algorithm : Burt Matrix")
text(Cb2, colnames(Z), pos=3, col=1)
abline(v=0, h=0)