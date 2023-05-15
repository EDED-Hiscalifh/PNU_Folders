chisq.plot<-function(x){
     n=dim(x)[[1]]
     p=dim(x)[[2]]
     S=cov(x)
     m=mahalanobis(x, colMeans(x), S)
     m=sort(m)
     id=seq(1, n)
     pt=(id-0.5)/n
     q=qchisq(pt, p)
plot(q, m, pch="*")
abline(0, 1)
retrurn(list(xbar, S))
}

iris
setosa = iris[1:50, 1:4]
chisq.plot(setosa)
     