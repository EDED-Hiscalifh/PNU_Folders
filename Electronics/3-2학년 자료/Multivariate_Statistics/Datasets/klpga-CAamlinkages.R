# AMCA : AM Linkages
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X<-Data1.3.2
X<-as.matrix(Data1.3.2)
선수<-rownames(X) 

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
#단일연결법
sinle=hclust(ds, method="single")
plot(sinle, labels=선수, hang=-1, main="(a) Sinle Linkage")
#완전연결법
complete=hclust(ds, method="complete")
plot(complete, labels=선수, hang=-1, main="(b) Complete Linkage")
#평균연결법
average=hclust(ds, method="average")
plot(average, labels=선수, hang=-1, main="(c) Average Linkage")
#와드연결법
ward=hclust(ds, method="ward")
plot(ward, labels=선수, hang=-1, main="(d) Ward Linkage")

