# K-Means CA for Public Utilites
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data5.3.1<-read.table("utility.txt", header=T)
X<-Data5.3.1[,-1]
Z<-scale(X)
company=Data5.3.1[,1]

# K-means Method
kmeans <- kmeans(Z, 4) # 4 cluster solution

# Append cluster assignment
cluster=data.frame(company,cluster=kmeans$cluster)
cluster
C1=cluster[(cluster[,2]==1),]
C1
C2=cluster[(cluster[,2]==2),]
C2
C3=cluster[(cluster[,2]==3),]
C3
C4=cluster[(cluster[,2]==4),]
C4

# Get cluster means 
aggregate(X,by=list(kmeans$cluster),FUN=mean)

# Determine number of clusters
 wss <- (nrow(Z)-1)*sum(apply(Z,2,var))
 for (i in 2:15) wss[i] <- sum(kmeans(Z, 
    centers=i)$withinss)
 plot(1:15, wss, type="b", xlab="Number of Clusters",
   ylab="Within groups sum of squares") 


