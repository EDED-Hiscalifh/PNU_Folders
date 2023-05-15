# Importing libraries 
library(tibble)
library(MVN)
library(MASS)
library(psych)
library(car)
library(caret)
library(FactoMineR)
library(factoextra)
library(ggdendro)
library(dplyr)

# Dataset Preparing 
df <- read.csv('Multivariate_Statistics/Datasets/Pokemon.csv')
head(df)

# Filter columns for analysis 
using_cols <- c('Name', 'HP', 'Attack', 'Defense', 'Sp..Atk', 'Sp..Def', 'Speed')
X <- df[, using_cols] %>% 
  column_to_rownames('Name')
head(X)
summary(X)

# Explore Descriptive Statistics 
corr <- cor(X)
corr

boxplot(X)

# Normality Test
result = mvn(X, mvnTest = 'mardia', multivariatePlot = "qq")
result

# Scaled Normality Test 
scaled_X <- scale(X, scale = T)

for (i in 1:dim(X)[2]) {
  col <- X[, i]
  Q1 <- quantile(col, probs = 0.2)
  Q3 <- quantile(col, probs = 0.8)
  IQR <- Q3 - Q1 
  upper <- Q3 + 1.5 * IQR
  lower <- Q1 - 1.5 * IQR 
  col[col > upper] <- upper 
  col[col < lower] <- lower
  bc <- boxcox(col ~ 1)
  lambda <- bc$x[which.max(bc$y)]
  scaled_X[, i] <- (col^lambda - 1)/lambda
}
result = mvn(scaled_X, mvnTest = 'mardia', multivariatePlot = "qq")
result

# FA
## Scree plot
R = round(cor(X), 3)
eigen.R = eigen(R)
V = round(eigen.R$vectors, 2)

gof = eigen.R$values/p*100
plot(eigen.R$values, type = 'b', main = 'Scree Graph', xlab = "Factor Number", ylab = "Eigenvalue")

## pcfa
pcfa <- principal(R, nfactors = 2, rotate = "none")
pcfa

L = pcfa$loadings[,1:2]
rownames(L) <- colnames(X)
colnames(L) <- c("PC1", "PC2")
lim <- range(pretty(L))
plot(L[,1], L[,2], main = "Plot of Factor Loadings : none ", xlab = "PC1", ylab = "PC2", xlim=lim, ylim=lim)
text(L[,1], L[,2], labels = rownames(L), cex = 0.8, col = "blue", abline(v=0, h=0))
arrows(0,0,L[,1],L[,2], col=2, code=2, length = 0.1)


## Factor Score
Z <- scale(X, scale = T)
pcfa <- principal(Z, nfactors = 2, rotate = 'varimax')
pcfa

## Residual Matrix
L=pcfa$loading[, 1:2]
round(L, 3)
Psi=pcfa$uniquenesses
round(Psi, 3)
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

lim<-range(pretty(L))
plot(L[,1], L[,2],main="(a) PC Factor Loadings : PC1 and PC2",  xlab="PC1", ylab="PC2",
     xlim=lim, ylim=lim)
text(L[,1], L[, 2], labels=rownames(L), cex=0.8, col="blue", pos=1)
abline(v=0, h=0)
arrows(0,0, L[,1], L[, 2], col=2, code=2, length=0.1)

fpc=pcfa$scores
round(fpc, 3)
lim<-range(pretty(fpc))
plot(fpc[,1], fpc[,2],main="(a) Factor Scores : PC1 and PC2",  xlab="PC1", ylab="PC2",
     xlim=lim, ylim=lim)
text(fpc[,1], fpc[,2], labels=rownames(fpc), cex=0.8, col="blue", pos=1)
abline(v=0, h=0)

biplot(pcfa$scores[,1:2], pcfa$loadings[,1:2], main="(a) Factor Scores : PC1 and PC2",  xlab="PC1", ylab="PC2")

# CA

p <- princomp(X, cor = TRUE)
pca_sc <- p$scores[, 1:2]
de <- as.matrix(dist(pca_sc, method = 'euclidean'))
de <- as.dist(de)

## 단일 연결법 
single = hclust(de, method = 'single')
plot(single, hang = -1, main = "Single Linkageｓ")

## 완전 연결법
complete = hclust(de, method = 'complete')
plot(complete, hang = -1, main = "Complete Linkage")

## 와드 연결법 
ward = hclust(de, method = "ward.D")
plot(ward, hang = -1, main = "Ward Linkage")

## K-means Clustering
tot_withinss <- c()
for (i in 1:20){
  set.seed(1004) # for reproducibility
  kmeans_cluster <- kmeans(de, centers = i, iter.max = 1000)
  tot_withinss[i] <- kmeans_cluster$tot.withinss}

plot(c(1:20), tot_withinss, type="b",  
     main="Optimal number of clusters", 
     xlab="Number of clusters", 
     ylab="Total within-cluster sum of squares")

kmeans <- kmeans(de, 7)
fviz_cluster(kmeans, data = de, geom = c("point"), ellipse.type = "euclid")
kmeans$cluster

Z <- scale(X, scale = T)
de <- as.matrix(dist(Z, method = 'euclidean'))

tot_withinss <- c()
for (i in 1:20){
  set.seed(1004) # for reproducibility
  kmeans_cluster <- kmeans(de, centers = i, iter.max = 1000)
  tot_withinss[i] <- kmeans_cluster$tot.withinss}

plot(c(1:20), tot_withinss, type="b",  
     main="Optimal number of clusters", 
     xlab="Number of clusters", 
     ylab="Total within-cluster sum of squares")

kmeans <- kmeans(Z, 7)
fviz_cluster(kmeans, data = de, geom = c("point"), ellipse.type = "euclid")
kmeans$cluster

# Compare Cluster and Types 
types <- df[, c('Name', 'Type.1', 'Type.2', 'Generation')]
types_cluster <- data.frame(cbind(types, kmeans$cluster))
rownames(types_cluster) <- 1:nrow(types_cluster)
head(types_cluster)
