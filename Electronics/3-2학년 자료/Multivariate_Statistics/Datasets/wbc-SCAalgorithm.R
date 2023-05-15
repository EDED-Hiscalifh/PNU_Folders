O<-matrix(c(91, 90, 51,
           150, 200, 155,
           109, 198, 172), byrow=T, nrow=3)
F <- O/sum(O)
	r <- apply(F,1,sum)
 	c <- apply(F,2,sum)
	Dr<- diag(1/sqrt(r))
	Dc<- diag(1/sqrt(c))
Y <- Dr%*%(F-r%*%t(c))%*%Dc
svd.Y <- svd(Y)
	U <- svd.Y$u 
	V <- svd.Y$v
	D <- diag(svd.Y$d)
Cr <- (Dr%*%U%*%D)[,1:2]
Cc <- (Dc%*%V%*%D)[,1:2]
	rownames(Cr) <- c("<45", "45-49", "50+")
	rownames(Cc) <- c("매달", "수시로", "안한다")

eig <- (svd.Y$d)^2 
	per <- eig/sum(eig)*100
	gof <- sum(per[1:2])
rbind(round(eig, 3),round(per, 3))
par(pty="s")
	lim <-range(pretty(Cr))
	plot(Cc[, 1:2], xlab="Dim1",ylab="Dim2",xlim=lim,ylim=lim,pch=15,col=2, main="SCA Biplot of WBC : SCA Algorithm")
	text(Cc[, 1:2],rownames(Cc),cex=0.8,col=2,pos=3)
	points(Cr[, 1:2],pch=16, col=4)
	text(Cr[, 1:2],rownames(Cr),cex=0.8,pos=3, col=4)
      abline(v=0,h=0)