#Clark distance from data matrix
X <- matrix(c(6,4,4,8,4,2,2,2),byrow=T, nrow=4)
	n <- nrow(X)
rownames(X) <- c("a","b","c","d")
# Scatter Plot
par(pty="s")
lim <- range(pretty(X))
plot(X[,1],X[,2],xlab="X1",ylab="X2",xlim=lim,ylim=lim,pch=15,col=2)
abline(v=0,h=0)
text(X[,1],X[,2],rownames(X),cex=1,col=2,pos=3)

# Clark¡¯s distance
D <- matrix(0,n,n)
for (i in 1:n) { for (j in 1:n) {
tmp <- as.matrix((X[i,]-X[j,])/(X[i,]+X[j,]))
d <- sqrt(t(tmp)%*%tmp)
D[i,j] <- d}}
round(D, 3)

#Metric MDS: Torgerson's Algorithm
	A <- -(D^2)/2
	J <- matrix(1,n,n)
	H <- diag(1,n)-J/n
	B <- H%*%A%*%H
	eg.B <- eigen(B)
	tmp <- sum(eg.B$values>=0)
	eig <- eg.B$values[1:tmp]
	per <- eig/sum(eig)*100 
	gof <- sum(per[1:2]) 
	V <- eg.B$vectors[,1:2]
	C <- V%*%diag(sqrt(eig[1:2]))
list(eig, gof)
# MDS Map
par(pty="s")
	lim <- range(pretty(C))
	plot(C[,1],C[,2],xlab="Dim1",ylab="Dim2",xlim=lim,ylim=lim,pch=15,col=2)
	abline(v=0,h=0)
	text(C[,1],C[,2],rownames(X),cex=1,col=2,pos=3)