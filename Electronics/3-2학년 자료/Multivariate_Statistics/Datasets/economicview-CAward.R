# AMCA : Ward Linkage
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-Data1.3.5[,-1]
X<-as.matrix(Data1.3.5[,-1])
기관=Data1.3.5[,1]

n<-nrow(X)
xbar<-t(X)%*%matrix(1,n,1)/n # 평균벡터
I<-diag(n)
J<-matrix(1,n,n)
H<-I-1/n*J                  # 중심화행렬
Y<-H%*%X                 # 중심화 자료행렬
S<-t(Y)%*%Y/(n-1)          # 공분산행렬 
D<-diag(1/sqrt(diag(S)))     # 표준편차행렬의 역
Z<-Y%*%D                # 표준화자료행렬
colnames(Z)<-colnames(X)


# 유클리드거리
de <- as.matrix(dist(X, method="euclidean"))
de <- as.dist(de)
round(de, 3)
ward=hclust(de, method="ward")
plot(ward, labels=기관, main="(a) Ward Linkage : Euclidean Distance")

# 표준화 유클리드거리
ds <- as.matrix(dist(Z, method="euclidean"))
ds <- as.dist(ds)
round(ds, 3)
wards=hclust(ds, method="ward")
plot(wards, labels=기관,  main="(b) Ward Linkage : Standardized Euclidean Distance")
)

# 마할라노비스거리
library(biotools)
dm<-D2.dist(X, S)
round(sqrt(dm), 3)
wardm=hclust(ds, method="ward")
plot(wardm, labels=기관, main="(c) Ward Linkage : Mahalanobis Distance")

