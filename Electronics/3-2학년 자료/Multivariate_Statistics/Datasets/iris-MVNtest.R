library("MVN")
iris
# MVN tests based on the Skewness and Kurtosis Ststistics
par(mfrow=c(1,3))
setosa = iris[1:50, 1:4] # Iris data only for setosa and four variables
versicolor = iris[51:100, 1:4] # Iris data only for versicolor and four variables
virginica = iris[101:150, 1:4] # Iris data only for virginica and four variables

result_setosa = mardiaTest(setosa, qqplot = TRUE)
result_versicolor = mardiaTest(versicolor, qqplot = TRUE)
result_virginica = mardiaTest(virginica, qqplot = TRUE)

result_setosa
result_versicolor
result_virginica

