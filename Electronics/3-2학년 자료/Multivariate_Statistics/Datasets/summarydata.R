setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.1.1<-read.table("3subjects.txt", header=T)
X<-Data1.1.1[,-1]

# Descriptive Statistics
summary(X)

# Covariance Matrix
cov(X)

# Correlation Matrix
cor(X)

# Multiple Scatter Plot
plot(X)

# Boxplot of 3 Subjects 
boxplot(X, xlab="3 Subjects", ylab="Exam Marks")

# Stem-and-leaf Plot
stem(X[,1])
stem(X[,2])
stem(X[,3])








