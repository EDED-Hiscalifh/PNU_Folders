library("MVN")
library(biotools)

#[Step 1] Iris flower data : setosa, versicolor, virginica
iris
attach(iris)
pairs(iris[1:4], pch=21, bg=c("red", "green", "blue")[unclass(iris$Species)])

#[Step 2] MVN tests based on the Skewness and Kurtosis Ststistics
setosa = iris[1:50, 1:4] # Iris data only for setosa and four variables
versicolor = iris[51:100, 1:4] # Iris data only for versicolor and four variables
virginica = iris[101:150, 1:4] # Iris data only for virginica and four variables
result_setosa = mardiaTest(setosa, qqplot = TRUE)
result_versicolor = mardiaTest(versicolor, qqplot = TRUE)
result_virginica = mardiaTest(virginica, qqplot = TRUE)
list(result_setosa, result_versicolor, result_virginica)

#[Step 3] Box's M-Test for Equlaity of Covariance Matrices
boxM(iris[, -5], iris[, 5])
S1=cov(setosa);S2=cov(versicolor);S3=cov(virginica)
Sp=(49*S1+49*S2+49*S3)/(150-3)
list(S1, S2, S3, Sp)

# [Step 4] Quadratic DA with 3 groups applying 
# resubstitution prediction and equal prior probabilities.
QDA=qda(Species~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width, 
              data=iris, prior=c(1,1,1)/3)
qcluster=predict(QDA, iris)$class
table(Species, qcluster)

# Total percent correct
mean(Species==qcluster)


# Discriminant Analysis Based on Mahalanobis Distance
disc <- D2.disc(iris[, -5], iris[, 5])
first10 <- iris[1:10, -5]
predict(disc, first10)
cluster=predict(disc, iris[, -5])$class
qct1=table(Species, cluster)
qct1

