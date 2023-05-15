data(iris)  
      setosa = iris[1:50, 1:4] # Iris data only for setosa
   # versicolor = iris[51:100, 1:4] # Iris data only for versicolor
   # virginica = iris[101:150, 1:4] # Iris data only for virginica 

# Chi-squre Plot for Checking MVN
     x= setosa  
     n=dim(x)[[1]]
     p=dim(x)[[2]]
     S=cov(x)
     xbar=colMeans(x)
     m=mahalanobis(x, xbar, S)
     m=sort(m)
     id=seq(1, n)
     pt=(id-0.5)/n
     q=qchisq(pt, p)
     plot(q, m, pch="*", xlab="Quantile", ylab="Ordered Squared Distance")
     abline(0, 1)
# Correlation Coefficient Test for Normailty
     rq=cor(cbind(q, m))[1,2]
     rq

