# Importing library and data 
library(ISLR)
data(Caravan)
dim(Caravan)
str(Caravan)
attach(Caravan)

# only 6% of people purchased caravan insurance.
summary(Purchase)
mean(Purchase=="Yes")

# Logistic regression 
g0 <- glm(Purchase~., data=Caravan, family="binomial")
summary(g0)

library(glmnet)
y <- Purchase
x <- as.matrix(Caravan[,-86])

# glmnet with cross validation
set.seed(123)
g1.cv <- cv.glmnet(x, y, alpha=1, family="binomial")
plot(g1.cv)

# Extract the value of lambda of model g1.cv 
g1.cv$lambda.min
g1.cv$lambda.1se

# Check coefficients 
coef1 <- coef(g1.cv, s="lambda.min")
coef2 <- coef(g1.cv, s="lambda.1se")
cbind(coef1, coef2)

# Degree of freedom 
sum(coef1!=0)-1
sum(coef2!=0)-1



# Standardize data so that mean=0 and variance=1.
X <- scale(Caravan[,-86])
apply(Caravan[,1:5], 2, var)
apply(X[,1:5], 2, var)

# Separate training sets and test sets
test <- 1:1000
train.X <- X[-test, ]
test.X <- X[test, ]
train.Y <- Purchase[-test]
test.Y <- Purchase[test]

library(class)

## Classification error rate of KNN
set.seed(1)

knn.pred <- knn(train.X, test.X, train.Y, k=1)
mean(test.Y!=knn.pred)
mean(test.Y!="No")
table(knn.pred, test.Y)

knn.pred=knn(train.X, test.X, train.Y, k=3)
table(knn.pred, test.Y)
mean(test.Y!=knn.pred)

knn.pred=knn(train.X, test.X, train.Y, k=5)
table(knn.pred, test.Y)
mean(test.Y!=knn.pred)

knn.pred=knn(train.X, test.X, train.Y, k=10)
table(knn.pred, test.Y)
mean(test.Y!=knn.pred)

library(ISLR)
data(Caravan)
attach(Caravan)
library(class)

set.seed(1234)
n <- nrow(Caravan) 
s <- sample(rep(1:3, length=n))
tran <- s==1
valid <- s==2 
test <- s==3 
K = 100 

X <- scale(Caravan[,-86])
y <- Caravan[,86]

train.X <- X[tran,]
valid.X <- X[valid,]
test.X <- X[test,]
train.y <- y[tran]
valid.y <- y[valid]
test.y <- y[test]

miss <- rep(0, K)
for (i in 1:K) { 
  knn.pred <- knn(train.X, valid.X, train.y, k=i)
  miss[i] <- mean(valid.y != knn.pred)
}
miss
wm <- which.min(miss)

miss_test <- knn(train.X, test.X, train.y, k=wm)
mean(test.y != miss_test)
