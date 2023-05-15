# Import library and data
data(iris)
library(e1071) 

# Train model 
g1 <- naiveBayes(Species ~ ., data=iris) 
g1 <- naiveBayes(iris[,-5], iris[,5])
pred <- predict(g1, iris[,-5]) 
table(pred, iris[,5]) 
mean(pred!=iris$Species) 

# Randomly separate training sets and test sets
set.seed(1234)
tran <- sample(nrow(iris), size=floor(nrow(iris)*2/3))

# Compute misclassification error for test sets
g2 <- naiveBayes(Species ~ ., data=iris, subset=tran)
pred2 <- predict(g2, iris)[-tran]
test <- iris$Species[-tran]
table(pred2, test)
mean(pred2!=test)

# Import dataset 
data(Default)

# Train-test split 
set.seed(1234)
n <- nrow(Default)
train <- sample(1:n, n*0.7)
test <- setdiff(1:n, train)

# train model and calculate missclassification rate 
g3 <- naiveBayes(default ~ ., data=Default, subset=train)
pred3 <- predict(g3, Default)[test]
table(pred3, Default$default[test])
mean(pred3!=Default$default[test])

# AUC of Naive Bayes 
library(ROCR)
label <- factor(default[test], levels=c("Yes","No"),
                labels=c("TRUE","FALSE"))
pred4 <- predict(g3, Default, type="raw")
preds <- prediction(pred4[test, 2], label)
performance(preds, "auc")@y.values