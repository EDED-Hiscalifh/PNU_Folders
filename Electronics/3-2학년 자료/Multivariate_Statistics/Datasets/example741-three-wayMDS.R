# 2-dimensional Objects' Configuration Coordinates
X=matrix(c(-0.5, -1.6, -1.3, -1.2, 1.6, 0.8), nrow=3, byrow=T)
id=c("o1", "o2", "o3")
# Subjects' Weighted Matrix
W=matrix(c(0.6, 0.4, 0.8, 0.2), nrow=2, byrow=T)
sub=c("s1", "s2")
win.graph()
par(mfrow=c(1,2))
lim<-c(-2, 2)

# Objects' Configuration Plot
plot(X, xlab="Dim1", ylab="Dim2",  xlim=lim, ylim=lim)
text(X,id, cex=0.8, pos=3)
lines(X[c(1:3,1),])
abline(v=0, h=0)
lim<-range(pretty(W))
plot(W, xlab="Dim1", ylab="Dim2", xlim=lim, ylim=lim)
text(W,sub, cex=0.8, pos=3)


# Subjects' Configrarion Coordinates
Y1=X%*%diag(sqrt(W[1,]))
Y2=X%*%diag(sqrt(W[2,]))

win.graph()
par(mfrow=c(1,2))
lim<-c(-2, 2)
# Subject 1 Weighted Configuration Plot
plot(Y1, type="o", xlab="Dim1", ylab="Dim2",  xlim=lim, ylim=lim)
text(Y1,id, cex=0.8, pos=3)
lines(Y1[c(1:3,1),])
abline(v=0, h=0)
# Subject 2 Weighted Configuration Plot
plot(Y2, type="o", xlab="Dim1", ylab="Dim2",  xlim=lim, ylim=lim)
text(Y2,id, cex=0.8, pos=3)
lines(Y2[c(1:3,1),])
abline(v=0, h=0)