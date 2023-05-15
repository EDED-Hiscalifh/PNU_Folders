setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Table1.1<-read.table("Table1-1.txt", header=T)
X<-as.matrix(Table1.1[, -1])
rownames(X)<-Table1.1[,1]
colnames(X)
Mechanics<-X[,1]
Algebra<-X[,2]
Statistics<-X[,3]

# Descriptive Statistics
library(psych)
describe(X)

# Covariance Matrix
cov(X)

# Correlation Matrix
cor(X)

# Multiple Scatter Plot
pairs(~Mechanics+Algebra+Statistics, data=X)

# Boxplot of 3 Subjects 
boxplot(Mechanics, Algebra,Statistics, names=colnames(X),
   xlab="3 Subjets", ylab="Exam Marks") 

# Stem-and-leaf Plot
stem(Mechanics)
stem(Algebra)
stem(Statistics)






