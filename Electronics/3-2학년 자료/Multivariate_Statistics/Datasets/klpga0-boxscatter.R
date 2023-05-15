setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga0.txt", header=T)
X<-Data1.3.2[,-1]
X5<-round(X[, 5]/max(X[,5]), 3)*100
X<-cbind(X[, -5], X5)
X
# Descriptive Statistics
summary(X)

# Covariance Matrix
cov(X)

# Correlation Matrix
cor(X)

# Multiple Scatter Plot
plot(X)

# Boxplot of 3 Subjects 
boxplot(X[,-5])

# Stem-and-leaf Plot
stem(X[,1])
stem(X[,2])
stem(X[,3])
