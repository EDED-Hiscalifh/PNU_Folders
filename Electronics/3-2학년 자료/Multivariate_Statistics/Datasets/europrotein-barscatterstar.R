setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.3<-read.table("europrotein.txt", header=T)
X<-Data1.3.3

# Multiple Scatter Plot
plot(X)

# Barplot of 25 Countries
X<-t(X)
par(las=2)
par(mar=c(8,8,4,2)) 
barplot(X, legend=rownames(X), horiz=TRUE)


# Star Plot
X<-scale(Data1.3.3)
stars(X,key.loc=c(0,2), full = FALSE)

