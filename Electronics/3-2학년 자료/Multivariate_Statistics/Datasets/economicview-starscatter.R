setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-Data1.3.5[, -1]

# Multiple Scatter Plot
plot(X)

# Star Plot
X<-scale(as.matrix(X[, c(1,2,3,5)]))
rownames(X)<-Data1.3.5[, 1]
stars(X,key.loc=c(8,2), full = FALSE)




