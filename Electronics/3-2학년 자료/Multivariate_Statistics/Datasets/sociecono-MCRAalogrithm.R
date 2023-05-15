setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
sociecono<-read.table("sociecono.txt", head=T) 
B<-as.matrix(sociecono)
colnames<-colnames(B)
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
lam
fit<-lam[1:2]/sum(lam)*100
rbind(round(lam[1:2], 3),round(fit, 3))
Cb<- Dc%*%V%*%Dl
Cb2=Cb[, 1:2]
rownames(Cb2)<-colnamesr
round(Cb2, 3)
par(pty="s")
lim<-range(pretty(Cb))
plot(Cb2, xlab="Dim1", ylab="Dim2", xlim=lim, ylim=lim, main="MCRA algorithm : Burt Matrix")
text(Cb2, colnames, col=2,  pos=3)
abline(v=0, h=0)
