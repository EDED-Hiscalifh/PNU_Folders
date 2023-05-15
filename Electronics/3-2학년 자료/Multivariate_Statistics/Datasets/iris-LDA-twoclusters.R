library("MVN")
iris
# MVN tests based on the Skewness and Kurtosis Ststistics

sts_vsc = iris[1:100, 1:5] # setosa and versicolor data for four variables
sts_vsc=sts_vsc[, c(1,2,5)]
attach(sts_vsc)
plot(sts_vsc, pch=unclass(Species), col=1:2)

attach(iris)
sts_vsc = iris[1:100, 1:5]
library(MASS)
ldahis(data= sts_vsc[, 1], g=sts_vsc[,3], type="density")
ld=lda(Species~Sepal.Length+Sepal.Width, data=iris)

ld
fit=lda(sts_vsc[, 1:2], sts_vsc[,3])