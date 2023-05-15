setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.3<-read.table("protein.txt", header=T)
X<-Data1.3.3[,-1]
rownames(X)<-Data1.3.3[,1]
colnames(X)
# Descriptive Statistics
summary(X)

# Covariance Matrix
cov(X)

# Correlation Matrix
cor(X)

# Multiple Scatter Plot
plot(X)

# Boxplot of 3 Subjects 
boxplot(X)