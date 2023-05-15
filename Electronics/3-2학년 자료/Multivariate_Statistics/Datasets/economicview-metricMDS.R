setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-scale(Data1.3.5[,-1], scale=T)
X<-as.matrix(Data1.3.5[,-1])
기관=Data1.3.5[,1]

#표준화 유클리드거리
D <- as.matrix(dist(X, method="euclidean"))

# Metric MDS
con<-cmdscale(D, k=2, eig=T)
con
x<-con$points[,1]
y<-con$points[,2]
#lim<-c(-max(abs(con$points)), max(abs(con$points)))
lim1<-c(-max(abs(x)), max(abs(x)))
lim2<-c(-max(abs(y)), max(abs(y)))
plot(x,y, xlab="Dim1", ylab="Dim2", xlim=lim1, ylim=lim2)
text(x,y, 기관, cex=0.8, pos=1)
abline(v=0, h=0)

