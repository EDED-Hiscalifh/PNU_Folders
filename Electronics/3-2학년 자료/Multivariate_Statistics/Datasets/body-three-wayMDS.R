# Body data: Three Way MDS(INDSCAL)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data7.4.1<-read.table("body.txt", header=T)
attach(Data7.4.1)
D<-as.matrix(Data7.4.1)
부위=colnames(D)
n=nrow(D)
d <- list(NULL)
for(i in 1:30){
   d[[i]] <- as.dist(D[(i*15-14):(i*15),])
}
d
library(smacof)

body.diag <- smacofIndDiff(d, ndim=3, constraint = "indscal") ## diagonally restricted weights
summary(body.diag)
w=matrix(unlist(body.diag$cweights), ncol=3, byrow=T)
w1=colSums(w[1:45, ])/15
w2=colSums(w[46:90, ])/15
w1;w2
body.diag$stress
C=body.diag$gspace
plot3d(body.diag)
C
win.graph()
par(mfrow=c(1,2))
lim<-range(pretty(C))
plot(C[, 1], C[,2], xlab="Dim1", ylab="Dim2", xlim=lim, ylim=lim)
text(C[, 1], C[,2],부위, cex=0.8, pos=3)
abline(v=0, h=0)
plot(C[, 1], C[,3], xlab="Dim1", ylab="Dim3", xlim=lim, ylim=lim)
text(C[, 1], C[,3],부위, cex=0.8, pos=3)
abline(v=0, h=0)
win.graph()
plot(body.diag, plot.type = "Shepard")
plot(body.diag, plot.type = "resplot")
