setwd("c:/R°ú ÇÔ²²ÇÏ´Â ´Ùº¯·®ÀÚ·áºÐ¼®/R_code_data")
Data1.1.5<-read.table("irisflower.txt", header=T)
X<-Data1.1.5[, -1]
# Box Plot
par(mfrow=c(2,2)) 
boxplot(²É¹ÞÄ§±æÀÌ~group, data=X, xlab="º×²É Á¾·ù", ylab="X1: ²É¹ÞÄ§ ±æÀÌ")
boxplot(²É¹ÞÄ§Æø~group, data=X, xlab="º×²É Á¾·ù", ylab="X2: ²É¹ÞÄ§ Æø")
boxplot(²ÉÀÙ±æÀÌ~group, data=X, xlab="º×²É Á¾·ù", ylab="X3: ²ÉÀÙ ±æÀÌ")
boxplot(²ÉÀÙÆø~group, data=X, xlab="º×²É Á¾·ù", ylab="X4: ²ÉÀÙ Æø")

# Multiple Scatter Plot
plot(X[,1:4], pch=unclass(X[,5]), col=1:3)

