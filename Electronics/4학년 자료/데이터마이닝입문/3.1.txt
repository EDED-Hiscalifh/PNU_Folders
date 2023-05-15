data(iris)
a <- subset(iris, Species == "setosa" | Species == "versicolor")
a$Species <- factor(a$Species)
str(a)

b <- glm(Species~Sepal.Length, data=a, family=binomial)
summary(b)
coef(b)
exp(coef(b)["Sepal.Length"])
confint(b, parm = "Sepal.Length")
exp(confint(b, parm = "Sepal.Length"))

fitted(b)[c(1:5, 96:100)]
predict(b, newdata=a [c(1, 50, 51, 100), ], type="response")

cdplot(Species~Sepal.Length, data=a)


plot(a$Sepal.Length, a$Species, xlab="Sepal.Length")
x=seq(min(a$Sepal.Length), max(a$Sepal.Length), 0.1)
lines(x, (1/(1+(1/exp(-27.831+5.140*x)))), type="l", col="red")
lines(x, 1+(1/(1+(1/exp(-27.831+5.140*x)))), type="l", col="red")



####################################
attach(mtcars)
str(mtcars)

glm.vs <- glm(vs~mpg+am, data=mtcars, family=binomial)
summary(glm.vs)

step.vs <- step(glm.vs, direction="backward")
summary(step.vs)

ls(glm.vs)
str(glm.vs)

anova(glm.vs, test="Chisq")
1-pchisq(18.327, 1)
1-pchisq(4.887, 1)