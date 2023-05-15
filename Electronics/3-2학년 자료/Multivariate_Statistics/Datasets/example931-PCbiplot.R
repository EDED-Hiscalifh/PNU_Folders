Y<-matrix(c(2, 2, -4,
             2, 1, -3,
             0, -1.5, 1.5,
            -1, -0.5, 1.5), byrow=T, nrow=4)
rownames(Y)<-c("a1", "a2", "a3", "a4")
colnames(Y)<-c("b1","b2","b3")
n<-nrow(Y)

win.graph()
biplot(princomp(Y), pc.biplot=T, xlab="Dim1", ylab="Dim2", 
             main="(b) PC Biplot : pc.biplot() ", cex=1.5, pch=16, col=1 )
abline(v=0,h=0)


# Biplot based on the Singular Value Decomposition
svd.Y <- svd(Y) 
U <- svd.Y$u    
V <- svd.Y$v 
D <- diag(svd.Y$d)
A <-  (sqrt(n-1)*U)[,1:2]
B <- (sqrt(1/(n-1))*V%*%D)[,1:2] 

rownames(A)<-rownames(Y)
rownames(B)<-colnames(Y)

# Godness-of-fit
eig <- (svd.Y$d)^2 
per <- eig/sum(eig)*100
gof <- sum(per[1:2])
list(per, gof)

win.graph()
# PC Biplot
lim<-range(pretty(A))
biplot(A,B, xlab="Dim1", ylab="Dim2", main="(a) PC Biplot: SVD",
                 xlim=lim,ylim=lim, cex=1.5, pch=16, col=1)
abline(v=0,h=0)

