# Install if needed by removing the #
# install.packages("tidyverse")
# install.packages("readxl")
install.packages("FactoMineR")
install.packages("factoextra")# Load Libraries

library(tidyverse)
library(readxl)
library(FactoMineR)
library(factoextra)
# Dataset Loading 
df <- read.table("Multivariate_Statistics/Datasets/trackrecord2005-men.txt", fileEncoding = "EUC-KR", header = T)
head(df)
dim(df)

X <- as.matrix(df)

# Problem1 =========================================================================================================================
p <- princomp(X, cor = TRUE)
pca_sc <- p$scores[, 1:2]
de <- as.matrix(dist(X, method = 'euclidean'))
de <- as.dist(de)

# Problem2 =========================================================================================================================

# 단일 연결법 
single = hclust(de, method = 'single')
plot(single, hang = -1, main = "Single Linkage")

# 완전 연결법
complete = hclust(de, method = 'complete')
plot(complete, hang = -1, main = "Complete Linkage")

# 와드 연결법 
ward = hclust(de, method = "ward.D")
plot(ward, hang = -1, main = "Ward Linkage")

# Problem 3 =========================================================================================================================

tot_withinss <- c()
for (i in 1:20){
  set.seed(1004) # for reproducibility
  kmeans_cluster <- kmeans(de, centers = i, iter.max = 1000)
  tot_withinss[i] <- kmeans_cluster$tot.withinss}

plot(c(1:20), tot_withinss, type="b",  
     main="Optimal number of clusters", 
     xlab="Number of clusters", 
     ylab="Total within-cluster sum of squares")

kmeans <- kmeans(de, 3)
fviz_cluster(kmeans, data = de, geom = c("point"), ellipse.type = "euclid")
kmeans$cluster

# Problem 4 =========================================================================================================================
Z <- scale(X, scale = T)

tot_withinss <- c()
for (i in 1:20){
  set.seed(1004) # for reproducibility
  kmeans_cluster <- kmeans(Z, centers = i, iter.max = 1000)
  tot_withinss[i] <- kmeans_cluster$tot.withinss}

plot(c(1:20), tot_withinss, type="b",  
     main="Optimal number of clusters", 
     xlab="Number of clusters", 
     ylab="Total within-cluster sum of squares")

kmeans <- kmeans(Z, 3)
fviz_cluster(kmeans, data = de, geom = c("point"), ellipse.type = "euclid")
kmeans$cluster
