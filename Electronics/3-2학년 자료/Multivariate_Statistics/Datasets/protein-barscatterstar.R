setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.3<-read.table("protein.txt", header=T)
X<-Data1.3.3[, -(1:2)]

# Multiple Scatter Plot
plot(X)

# Barplot of 25 Countries
X<-as.matrix(Data1.3.3[, -(1:2)])
rownames(X)<-Data1.3.3[,2]
X<-t(X)
par(las=2)
par(mar=c(8,8,4,2)) 
barplot(X, legend=rownames(X), horiz=TRUE)


# Star Plot
X<-scale(as.matrix(Data1.3.3[, -(1:2)]))
rownames(X)<-Data1.3.3[, 2]
stars(X,key.loc=c(0,2), full = FALSE)

