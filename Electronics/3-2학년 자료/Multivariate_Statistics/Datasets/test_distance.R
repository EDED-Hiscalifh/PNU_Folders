#setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
#Data1.1.1<-read.table("3subjects.txt", header=T)
#X<-Data1.1.1[,-1]
#X<-as.matrix(Data1.1.1[,-1])
s1=c(90, 80)
s2=c(80,90)
s3=c(75, 80)
s4=c(70,70)
s5=c(65, 80)
X<-as.matrix(rbind(s1, s2, s3, s4, s5))
X
n<-nrow(X)
xbar<-t(X)%*%matrix(1,n,1)/n # 평균벡터
I<-diag(n)
J<-matrix(1,n,n)
H<-I-1/n*J                # 중심화행렬
Y<-H%*%X               # 중심화 자료행렬
S<-t(Y)%*%Y/(n-1)         # 공분산행렬 
D<-diag(1/sqrt(diag(S)))    # 표준편차행렬의 역
Z<-Y%*%D          # 표준화자료행렬
colnames(Z)<-colnames(X)

# 유클리드거리
de <- as.matrix(dist(X, method="euclidean", diag=F))
de <- as.dist(de)
round(de, 3)


# 표준화 유클리드거리
ds <- as.matrix(dist(Z, method="euclidean", diag=F))
ds <- as.dist(ds)
round(ds, 3)


# 마할라노비스거리
library(biotools)
dm<-D2.dist(X, S)
round(sqrt(dm), 3)


# 시티블럭거리
dc <- as.matrix(dist(X, method="manhattan", diag=F))
dc <- as.dist(dc)
round(dc, 3)


