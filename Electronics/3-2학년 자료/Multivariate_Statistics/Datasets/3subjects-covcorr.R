setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.1.1<-read.table("3subjects.txt", header=T)
X<-Data1.1.1
X
X<-as.matrix(X)              # 자료행렬
n<-nrow(X)
xbar<-t(X)%*%matrix(1,n,1)/n # 평균벡터
I<-diag(n)
J<-matrix(1,n,n)
H<-I-1/n*J                # 중심화행렬
Y<-H%*%X                  # 중심화 자료행렬
S<-t(Y)%*%Y/(n-1)         # 공분산행렬 
D<-diag(1/sqrt(diag(S)))  # 표준편차행렬의 역
Z<-H%*%X%*%D              # 표준화자료행렬
colnames(Z)<-colnames(X)
R<-t(Z)%*%Z/(n-1)         # 상관행렬
R_S<-D%*%S%*%D           # 상관행렬과 공분산행렬의 대수적 관계
detS<-det(S)              # 일반화분산과 총분산
detR<-det(R)
trS<-sum(diag(S))
trR<-sum(diag(R))
# 결과 출력
X; xbar; Y; Z; S; R; detS; trS; detR; trR
