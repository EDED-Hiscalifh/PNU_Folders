setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-Data1.3.5[, -1]
X

# Multiple Scatter Plot
plot(X)

# Covariance Matrix
cov(X)

# Correlation Matrix
cor(X)


# Bar Plot
X<-scale(as.matrix(Data1.3.5[, -c(1, 5, 7, 9, 10, 11)]), scale=F)
rownames(X)<-Data1.3.5[, 1]
X<-t(X)
par(las=2)
par(mar=c(10,8,4,2)) 
barplot(X, legend=rownames(X), horiz=TRUE)


# Scatter Plot
par(pty="s")
par(mfrow=c(2,2))
plot(X[,c(2,1)], type="n")
text(X[,c(2,1)], labels=as.character(rownames(X)), cex=0.7, pos=3)
abline(v=0, h=0)


plot(X[, c(3,1)], type="n")
text(X[, c(3,1)], labels=as.character(rownames(X)), cex=0.7, pos=3)
abline(v=0, h=0)


X<-scale(as.matrix(X))

X<-scale(as.matrix(Data1.3.5[, -1]))

# Dissimilarity Matrix
X<-scale(as.matrix(X))
m <- as.matrix(dist(X, method="euclidean", diag=T))
m
d <- as.dist(m)
d
stopifnot(d == dist(m))



# Descriptive Statistics
summary(X)

# Covariance Matrix
cov(X)

# Correlation Matrix
cor(X)

# Multiple Scatter Plot
plot(X)

X<-scale(X)
# Boxplot of 3 Subjects 
boxplot(X, xlab="기관", ylab="경제 전망")

