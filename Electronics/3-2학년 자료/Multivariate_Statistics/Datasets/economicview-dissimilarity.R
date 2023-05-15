# Dissimilarity Matrix from Raw Data
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-Data1.3.5[, -1]
X<-scale(as.matrix(X))
m <-as.matrix(dist(X, method="euclidean", diag=T))
D<-round(m, 3)
D

# Euclidean Distance
n<-nrow(X)
D<-matrix(0, n, n)
for(i in 1:n){
   for(j in 1:n){
     D[i,j]<-sqrt(sum((X[i,]-X[j,])^2))
   }
}
D<-round(D, 3)
D


# Dissimilarity Matrix from Binary Data
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.7<-read.table("economicviewbinary.txt", header=T)
X<-Data1.3.7[, -1]
rownames(X)<-Data1.3.7[, 1]
n<-nrow(X)


# Dissimilarity Matrix from Binary Data
rownames(X)<-Data1.3.7[, 1]
m <-as.matrix(dist(X, method="euclidean", diag=T))
D<-round(m^2, 3)/10
D


# Squared Euclidean Distance from Binary Data
D<-matrix(0, n, n)
for(i in 1:n){
   for(j in 1:n){
     D[i,j]<-(sum((X[i,]-X[j,])^2))
   }
}
D<-D/10
D
# Dissimilarity Matrix from Binary Data
D<-1-X%*%t(X)/10
D





