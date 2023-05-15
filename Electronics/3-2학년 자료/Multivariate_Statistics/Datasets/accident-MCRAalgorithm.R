setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
accident<-read.table("accident.txt", head=T) 
B<-as.matrix(accident)
rownames(B)<-colnames(B)
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
rownames(Cb)<-colnames(B)
Cb2=Cb[, 1:2]
Cb2
limy<-range(pretty(Cb2))
limx<-c(-1, 1)
plot(Cb2, xlab="Dim1", ylab="Dim2", xlim=limx, ylim=limy, main="MCRA algorithm : Burt Matrix")
text(Cb2, rownames(Cb), col=1,  pos=3)
abline(v=0, h=0)