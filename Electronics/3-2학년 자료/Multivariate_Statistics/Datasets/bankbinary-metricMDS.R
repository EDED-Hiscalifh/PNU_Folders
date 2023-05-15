#[보기 7.2.5] Bank data의 계량형 MDS  
# Dissimilarity Matrix from Binary Data
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data7.2.2<-read.table("bankbinary.txt", header=T)
X<-Data7.2.2[, -1]
은행<-Data7.2.2[, 1]
n<-nrow(X)
p<-ncol(X)

# Dissimilarity Matrix from Binary Data
m <-as.matrix(dist(X, method="euclidean", diag=T))
D<-round(m^2, 3)/p

# Metric MDS
win.graph()
con<-cmdscale(D, k=2, eig=T)
con
x<-con$points[,1]
y<-con$points[,2]
lim<-c(-max(abs(con$points)), max(abs(con$points)))
plot(x,y, xlab="Dim1", ylab="Dim2", xlim=lim, ylim=lim)
text(x,y,은행, cex=0.8, pos=1)
abline(v=0, h=0)