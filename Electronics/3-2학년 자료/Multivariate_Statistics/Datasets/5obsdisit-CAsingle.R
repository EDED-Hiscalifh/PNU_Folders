# Hierachical Cluster Analysis 
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
D<-as.dist(read.table("5obsdist.txt", header=T))
single=hclust(D, method="single") # Single Linkage 
plot(single, hang=-1, main="Single Likage Dendrogram")