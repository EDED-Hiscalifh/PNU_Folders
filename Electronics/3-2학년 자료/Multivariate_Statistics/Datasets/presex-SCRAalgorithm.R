# Simple CRA : Two-Way Table
O<-matrix(c(81, 68,  60,  38,
            24, 26,  29,  14,
            18, 41,  74,  42,
            36, 57, 161, 157), byrow=T, nrow=4)
F <- O/sum(O)
	r <- apply(F,1,sum)
 	c <- apply(F,2,sum)
	Dr<- diag(1/sqrt(r))
	Dc<- diag(1/sqrt(c))
r;c;Dr;Dc
cF<- F-r%*%t(c)
Y <- Dr%*%(cF)%*%Dc
svd.Y <- svd(Y)
	U <- svd.Y$u 
	V <- svd.Y$v
	D <- diag(svd.Y$d)
A <- (Dr%*%U%*%D)[,1:2]
B <- (Dc%*%V%*%D)[,1:2]
	rownames(A) <- c("매우나쁨", "거의나쁨", "조금나쁨", "전혀안나쁨")
	rownames(B) <- c("매반", "반대", "찬성", "매찬")
A;B
eig <- (svd.Y$d)^2 
	per <- eig/sum(eig)*100
	gof <- sum(per[1:2])
rbind(round(eig, 3),round(per, 3))

par(pty="s")
	lim <-range(pretty(A))
	plot(B[, 1:2], xlab="Dim1",ylab="Dim2",xlim=lim,ylim=lim,pch=15,col=2, main="SCRA Algorithm : 이원분할표")
	text(B[, 1:2],rownames(B),cex=0.8,col=2,pos=3)
	points(A[, 1:2],pch=16, col=4)
	text(A[, 1:2],rownames(A),cex=0.8,pos=3, col=4)
      abline(v=0,h=0)
