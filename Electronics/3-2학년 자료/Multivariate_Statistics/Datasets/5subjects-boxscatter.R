setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.1<-read.table("5subjects.txt", header=T)
X<-Data1.3.1[, -1]

# Multiple Scatter Plot
plot(X)

# Box Plot
boxplot(X)

