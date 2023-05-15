
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("skull.txt", header=T)
Z=as.matrix(Data1.3.2)
Y=Z
n <- nrow(X) 
rownames(X)
colnames(X)
joinnames=c(rownames(X),colnames(X))


# Biplot based on the Singular Value Decomposition
svd.Y <- svd(Y) 
U <- svd.Y$u    
V <- svd.Y$v 
D <- diag(svd.Y$d)
G <- (sqrt(n-1)*U)[,1:2]
H <- (sqrt(1/(n-1))*V%*%D)[,1:2] 
C<- rbind(G, H)
rownames(G)<-rownames(X)
rownames(H)<-colnames(X)
rownames(C)<-joinnames

# Godness-of-fit
eig <- (svd.Y$d)^2 
per <- eig/sum(eig)*100
gof <- sum(per[1:2])

# Biplots
par(pty="s")

biplot(G,H, xlab="1st PC",ylab="2nd PC", main="PC Biplot ",
                 xlim=lim2,ylim=lim2,cex=0.8,pch=16)
abline(v=0,h=0)
