# CVM based on the holdout(jacknife)prediction : Evaluating Discriminant Function
library(MASS)
# Riding mower Data on two variables(x1.Income x2.Lotsize) 
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
ridingmower<-read.table("ridingmower.txt", header=T)
attach(ridingmower)
# Linear DA
LDA=lda(pop~x1.Income + x2.Lotsize, data=ridingmower, CV=TRUE)
# Confusion Table 
confusion_ridingmower=table(ridingmower$pop, LDA$class)
# Expected actual error rate : EAER
EAER=(1-sum(diag(prop.table(confusion_ridingmower))))*100
list(confusion_ridingmower, EAER)

# Iris flower data : setosa, versicolor, virginica
data(iris)
attach(iris)

# Quadratic DA with 3 groups applying 
# holdout(jacknife)prediction and equal prior probabilities.
QDA=qda(Species~ Sepal.Length+Sepal.Width+Petal.Length+Petal.Width, 
              data=iris, prior=c(1,1,1)/3, CV=TRUE)
# Confusion Table 
confusion_iris=table(iris$Species, QDA$class)
# Expected actual error rate : EAER
EAER=(1-sum(diag(prop.table(confusion_iris))))*100
list(confusion_iris, EAER)




