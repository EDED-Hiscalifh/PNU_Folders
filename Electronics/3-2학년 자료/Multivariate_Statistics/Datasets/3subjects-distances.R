setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.1.1<-read.table("3subjects.txt", header=T)
X<-as.matrix(Data1.1.1)
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
de <- as.matrix(dist(X, method="euclidean"))
de <- as.dist(de)
round(de, 3)


# 표준화 유클리드거리
ds <- as.matrix(dist(Z, method="euclidean"))
ds <- as.dist(ds)
round(ds, 3)


# 마할라노비스거리
library(biotools)
dm<-D2.dist(X, S)
round(sqrt(dm), 3)


# 시티블럭거리
dc <- as.matrix(dist(X, method="manhattan"))
dc <- as.dist(dc)
round(dc, 3)


