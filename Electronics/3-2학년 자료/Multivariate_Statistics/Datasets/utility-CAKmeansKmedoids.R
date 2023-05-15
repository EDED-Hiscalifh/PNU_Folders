# K-Means & K-Medoids(Partitioning Around Medoids)CA for Public Utilities
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data5.3.1<-read.table("utility.txt", header=T)
X<-Data5.3.1[,-1]
Z<-scale(X)
company=Data5.3.1[,1]

# K-means Method
kmeans <- kmeans(Z, 4) # 4 cluster solution
cluster=data.frame(company,cluster=kmeans$cluster)
C1=cluster[(cluster[,2]==1),]
C2=cluster[(cluster[,2]==2),]
C3=cluster[(cluster[,2]==3),]
C4=cluster[(cluster[,2]==4),]
C1;C2;C3;C4
# Get cluster means 
aggregate(X,by=list(kmeans$cluster),FUN=mean)

# K-medoids Method
library(cluster)
kmedoids <- pam(Z, 4, metric="euclidean") # 4 cluster solution
cluster=data.frame(company,cluster=kmedoids$cluster)
cluster
C1=cluster[(cluster[,2]==1),]
C2=cluster[(cluster[,2]==2),]
C3=cluster[(cluster[,2]==3),]
C4=cluster[(cluster[,2]==4),]
C1;C2;C3;C4
# Get cluster means 
aggregate(X,by=list(kmedoids$cluster),FUN=mean) 


