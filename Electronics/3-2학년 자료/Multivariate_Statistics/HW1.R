# Library and Datasets 
library(dslabs)
library(tidyr)
library(MVN)

ev_dataset <- read.table("Multivariate_Statistics/economicview.txt", encoding = "EUC-KR", header = T)

# View datasets 
str(ev_dataset)
X = ev_dataset[, -1]

# Q1. Find mean vector, covariance matrix S and Correlation matrix R

# mean vector
mean_vector <- colMeans(X)
mean_vector

# covariance matrix S
matrix_S <- cov(X)
matrix_S

# Correlation matrix R 
matrix_R <- cor(X)
matrix_R 

# Q2. Apply methods used in [R-code 1.3.1] and [R-code 1.3.2] and Explain their results

# R-code 1.3.2
summary(X)
cov(X)
cor(X)
plot(X)
boxplot(X)

# Q3. Between centering & standardizing, which one is more appropriate for the data? 

# centering 

centering_matrix <- scale(X, scale = F)
centering_matrix
boxplot(centering_matrix)

# standardizing 

standardizing_matrix <- scale(X, scale = T)
standardizing_matrix
boxplot(standardizing_matrix)

# Q4. Check the multivariate normality based on Chi-square plot and Mardia's test.

# Chi-square
n = dim(X)[[1]]
p = dim(X)[[2]]
S = cov(X)
xbar = colMeans(X)
m = mahalanobis(X, xbar, S)
m = sort(m)
id = seq(1, n)
pt = (id-0.5)/n
q = qchisq(pt, p)
plot(q, m, pch = "*", xlab = "Quantile", ylab = "Ordered Squared Distance")
abline(0, 1)

# MVN test
result = mvn(X, mvnTest = "mardia", multivariatePlot = "qq")
result

