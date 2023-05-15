# 10 European Cities’s Distances
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data7.2.1<-read.table("eurodist.txt", header=T)
D<-Data7.2.1
도시=colnames(D)
n<-nrow(D)

# Metric MDS
win.graph()
con<-cmdscale(D, k=2, eig=T)
con
x<-con$points[,1]
y<-con$points[,2]
lim1<-c(-max(abs(x)), max(abs(x)))
lim2<-c(-max(abs(y)), max(abs(y)))

plot(x,y, xlab="Dim1", ylab="Dim2", xlim=lim1, ylim=lim2)
text(x,y,도시, cex=0.8, pos=3)
abline(v=0, h=0)