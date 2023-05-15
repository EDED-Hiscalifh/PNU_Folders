# AMCA : Ward Linkage for US Public Utilities
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data5.3.1<-read.table("utility.txt", header=T)
X<-Data5.3.1[,-1]
X<-as.matrix(Data5.3.1[,-1])
회사=Data5.3.1[,1]

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

# 표준화 유클리드거리
ds <- as.matrix(dist(Z, method="euclidean"))
ds <- as.dist(ds)
round(ds, 3)
ward=hclust(ds, method="ward")
plot(ward, labels=회사, hang=-1,  main=" Ward Linkage : Standardized Euclidean Distance")

