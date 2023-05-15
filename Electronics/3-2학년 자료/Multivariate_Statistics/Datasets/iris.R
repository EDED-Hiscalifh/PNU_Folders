setwd("c:/R°ú ÇÔ²²ÇÏ´Â ´Ùº¯·®ÀÚ·áºÐ¼®/R_code_data")
Data1.1.5<-read.table("irisflower.txt", header=T)
X<-Data1.1.5

# Box Plot
par(mfrow=c(2,2)) 
boxplot(X1~group, data=X, xlab="º×²É Á¾·ù", ylab="X1: ²É¹ÞÄ§ ±æÀÌ")
boxplot(X2~group, data=X, xlab="º×²É Á¾·ù", ylab="X2: ²É¹ÞÄ§ Æø")
boxplot(X3~group, data=X, xlab="º×²É Á¾·ù", ylab="X3: ²ÉÀÙ ±æÀÌ")
boxplot(X4~group, data=X, xlab="º×²É Á¾·ù", ylab="X4: ²ÉÀÙ Æø")

# Multiple Scatter Plot
plot(X[,1:4], pch=unclass(X[,5]), col=1:3)



#scatter plot with group
par(pty="s")
par(mfrow=c(2, 3))
plot(X[,1],X[,2], pch=unclass(X[,5]),  ylab="X1: ²É¹ÞÄ§ ±æÀÌ", xlab="X2: ²É¹ÞÄ§ Æø")
 )  # different symbol
legend("bottomright", legend=levels(X[,5]), pch=c(1:3))

plot(X[,1],X[,3], pch=unclass(X[,5]),  ylab="X1: ²É¹ÞÄ§ ±æÀÌ", xlab="X3: ²ÉÀÙ ±æÀÌ")
 )  # different symbol
legend("bottomright", legend=levels(X[,5]), pch=c(1:3))
plot(X[,1],X[,4], pch=unclass(X[,5]),  ylab="X1: ²É¹ÞÄ§ ±æÀÌ", xlab="X4: ²ÉÀÙ Æø")
 )  # different symbol
legend("bottomright", legend=levels(X[,5]), pch=c(1:3))
plot(X[,2],X[,3], pch=unclass(X[,5]),  ylab="X2: ²É¹ÞÄ§ Æø", xlab="X3: ²ÉÀÙ ±æÀÌ")
 )  # different symbol
legend("bottomright", legend=levels(X[,5]), pch=c(1:3))
plot(X[,2],X[,4], pch=unclass(X[,5]),  ylab="X2: ²É¹ÞÄ§ Æø", xlab="X4: ²ÉÀÙ Æø")
 )  # different symbol
legend("bottomright", legend=levels(X[,5]), pch=c(1:3))
plot(X[,3],X[,4], pch=unclass(X[,5]),  ylab="X3: ²ÉÀÙ ±æÀÌ", xlab="X4: ²ÉÀÙ Æø")
 )  # different symbol
legend("bottomright", legend=levels(X[,5]), pch=c(1:3))


#scatter plot
library(lattice)
par(mfrow=c(2, 4))
xyplot(X1 ~ X2 | group, data = X) 
xyplot(X1 ~ X3 | group, data = X) 
xyplot(X1 ~ X4 | group, data = X) 
xyplot(X2 ~ X3 | group, data = X) 
xyplot(X2 ~ X4 | group, data = X) 
xyplot(X3 ~ X4 | group, data = X) 



library(ggplot2)
par(pty="s")
par(mfrow=c(3,2))
ggplot(X, aes(x = X1, y = X2, colour = group)) +
  geom_point() +
  facet_wrap( ~ group)
ggplot(X, aes(x = X1, y = X3, colour = group)) +
  geom_point() +
  facet_wrap( ~ group)
ggplot(X, aes(x = X1, y = X4, colour = group)) +
  geom_point() +
  facet_wrap( ~ group)
ggplot(X, aes(x = X2, y = X3, colour = group)) +
  geom_point() +
  facet_wrap( ~ group)
ggplot(X, aes(x = X2, y = X4, colour = group)) +
  geom_point() +
  facet_wrap( ~ group)
ggplot(X, aes(x = X3, y = X4, colour = group)) +
  geom_point() +
  facet_wrap( ~ group)






rownames(X)<-Data1.1.1[,1]
colnames(X)

data(iris)
iris

X<-iris[, -5]
X
plot(X, color=Species)

boxplot(X)


library(MVT)
data(examScor)
examScor
