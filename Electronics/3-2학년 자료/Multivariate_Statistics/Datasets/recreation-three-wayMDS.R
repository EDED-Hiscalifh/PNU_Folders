# Recreation data: Three Way MDS(INDSCAL)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data7.4.2<-read.table("recreation.txt", fill=T, header=T)
attach(Data7.4.2)
D<-as.matrix(Data7.4.2)
recreat=colnames(D)
d <- list(NULL)
for(i in 1:10){
   d[[i]] <- as.dist(D[(i*15-14):(i*15),])
}
d
library(smacof)
recreat.diag <- smacofIndDiff(d, ndim=2, constraint = "indscal")
summary(recreat.diag)
# Dimension Coefficients
w1=matrix(unlist(recreat.diag$cweights), ncol=2, byrow=T)
w=array(dim=c(10,2))
for(i in 1:10){
   w[i,] <- colSums(w1[(i*2-1):(i*2),])
}
stress=recreat.diag$stress
C=recreat.diag$gspace
list(C, w, stress)
win.graph()
lim<-range(pretty(C))
plot(C[, 1], C[,2], xlab="Dim1", ylab="Dim2", xlim=lim, ylim=lim)
text(C[, 1], C[,2],recreat, cex=0.8, pos=3)
abline(v=0, h=0)
win.graph()
par(mfrow=c(1,2))
plot(body.diag, plot.type = "Shepard")
plot(body.diag, plot.type = "resplot")
