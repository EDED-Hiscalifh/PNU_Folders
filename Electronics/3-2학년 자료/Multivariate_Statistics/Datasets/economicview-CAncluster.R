# K-Means & K-Medoids(Partitioning Around Medoids)CA for Economic Views
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-Data1.3.5[,-1]
X<-as.matrix(Data1.3.5[,-1])
Z<-scale(X)
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

# 표준화 유클리드거리
ds <- as.matrix(dist(Z, method="euclidean"))
ds <- as.dist(ds)
round(ds, 3)

library("NbClust")
ncluster<-NbClust(Z, distance="euclidean", min.nc = 2, max.nc = 8,
 method = "kmeans", index = "alllong", alphaBeale = 0.1)

# K-means Method
kmeans <- kmeans(Z, 4) # 4 cluster solution
cluster=data.frame(기관,cluster=kmeans$cluster)
C1=cluster[(cluster[,2]==1),]
C2=cluster[(cluster[,2]==2),]
C3=cluster[(cluster[,2]==3),]
C4=cluster[(cluster[,2]==4),]
C1;C2;C3;C4

# Get cluster means 
aggregate(X, by=list(kmeans$cluster),FUN=mean)

# K-medoids Method
library(cluster)
kmedoids <- pam(Z, 4, metric="euclidean") # 4 cluster solution
cluster=data.frame(기관,cluster=kmedoids$cluster)
C1=cluster[(cluster[,2]==1),]
C2=cluster[(cluster[,2]==2),]
C3=cluster[(cluster[,2]==3),]
C4=cluster[(cluster[,2]==4),]
C1;C2;C3;C4
# Get cluster means 
aggregate(X,by=list(kmedoids$cluster),FUN=mean) 