# Library and Datasets 
library(dslabs)
library(tidyr)
library(dplyr)
library(MVN)

# Dataset Loading 
X <- read.csv("Multivariate_Statistics/Stores.csv")

# Dataset View 
head(X)
str(X)
dim(X)

# Basic Statistics
summary(X)

# Divide dataset by Gender 
X_normal <- X[ , -1]

# Normality Test 
library(MVN)
result_normality = mvn(X_normal, mvnTest = "mardia", multivariatePlot = "qq")
result_normality

# Plotting
cov(X_normal)
cor(X_normal)
plot(X_normal)

?quantile
# Binning Store_Sales and plot 
q_25 <- quantile(X[, 'Store_Sales'], c(.25, .5, .75))[1]
q_5 <- quantile(X[, 'Store_Sales'], c(.25, .5, .75))[2]
q_75 <- quantile(X[, 'Store_Sales'], c(.25, .5, .75))[3]
X_bin_normal <- X_normal %>%
                  mutate(Store_Sales_bin = case_when(
                    Store_Sales <= q_25 ~ 'Low', 
                    Store_Sales > q_25 & Store_Sales < q_5 ~ 'Low-Mid',
                    Store_Sales >= q_5 & Store_Sales < q_75 ~ 'Mid_High',
                    Store_Sales >= q_75 ~ 'High'
                  )) %>%
                  select(Store_Area, Items_Available, Daily_Customer_Count, Store_Sales_bin)
plot(X_bin_normal[, 1:3], pch = '+', col = 1:4)

par(mfrow=c(2,2))
boxplot(Store_Area~Store_Sales_bin, data = X_bin_normal, xlab = "매장 소득", ylab = "매장 면적")
boxplot(Items_Available~Store_Sales_bin, data = X_bin_normal, xlab = "매장 소득", ylab = "가용 물품 수")
boxplot(Daily_Customer_Count~Store_Sales_bin, data = X_bin_normal, xlab = "매장 소득", ylab = "일별 방문 수")
