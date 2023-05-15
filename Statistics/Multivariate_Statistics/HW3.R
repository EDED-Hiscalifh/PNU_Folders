# Problem (1) ----------------------------------------------------------------------------------------------------------
# Data Matrix X 
df_3_5 = read.table("Multivariate_Statistics/Datasets/censustract.txt", fileEncoding = "EUC-KR", header = T)
X = df_3_5

rownames <- rownames(X)
colnames <- colnames(X)

# num cols
p = ncol(X)
n = nrow(X)

# Correlation Matrix 
R = round(cor(X), 3)
R

# Spectral Decomposition
eigen.R = eigen(R)
round(eigen.R$values, 2)
V = round(eigen.R$vectors, 2)

# Number of factors : m
gof = eigen.R$values/p*100
round(gof,3)
plot(eigen.R$values, type = 'b', main = 'Scree Graph', xlab = "Factor Number", ylab = "Eigenvalue")

# Problem(2) ----------------------------------------------------------------------------------------------------------------
# Factor Loadings and Communality
V2 = V[,1:2] 
L = V2%*%diag(sqrt(eigen.R$values[1:2]))
rownames(L) <- colnames(X)
colnames(L) <- c("PC1", "PC2")
round(L, 3)
round(diag(L%*%t(L)), 3)

# Specific Variance : Psi
Psi = diag(R-L%*%t(L))
round(Psi, 3)

# Residual Matrix 
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

# PCFA using the principal()
library(psych)
pcfa <- principal(R, nfactors = 2, rotate = "none")
pcfa
round(pcfa$values, 2)
gof = pcfa$values/p*100
round(gof, 3)
round(pcfa$residual, 2)

# Factor Loadings Plot
L = pcfa$loadings[,1:2]
rownames(L) <- colnames(X)
colnames(L) <- c("PC1", "PC2")
lim <- range(pretty(L))
plot(L[,1], L[,2], main = "Plot of Factor Loadings : none ", xlab = "PC1", ylab = "PC2", xlim=lim, ylim=lim)
text(L[,1], L[,2], labels = rownames(L), cex = 0.8, col = "blue", abline(v=0, h=0))
arrows(0,0,L[,1],L[,2], col=2, code=2, length = 0.1)

round(pcfa$values, 3)
gof=pcfa$values/p*100 # Goodness-of fit
round(gof, 3)

# Problem(3) ---------------------------------------------------------------------------------------------------------
# Factor Score
library(psych)
Z <- scale(X, scale = T)
pcfa <- principal(Z, nfactors = 2, rotate = 'varimax')
pcfa

# Residual Matrix
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


# Problem(4) ----------------------------------------------------------------------------------------------------
# MLFA 
mlfa <- factanal(covmat = R, factors = 2, rotation = "none")
mlfa

# Residual Matrix 
L = mlfa$loadings[,1:2]
Psi = mlfa$uniquenesses
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

# Factor Loadings Plot
lim <- range(pretty(L))
plot(L[,1], L[,2], main = "Plot of Factor Loadings : none ", xlim=lim, ylim=lim)
text(L[,1], L[,2], labels = rownames(L), cex = 0.8, col = "blue", abline(v=0, h=0))
arrows(0,0,L[,1],L[,2], col=2, code=2, length = 0.1)

library(psych)
Z <- scale(X, scale = T)
mlfa <- factanal(Z, factors = 2, rotation = 'varimax', score = "regression")
mlfa

# Residual Matrix
L=mlfa$loading[, 1:2]
round(L, 3)
Psi=mlfa$uniquenesses
round(Psi, 3)
Rm = R-(L%*%t(L) + diag(Psi))
round(Rm, 3)

lim<-range(pretty(L))
plot(L[,1], L[,2],main="(a) ML Factor Loadings : PC1 and PC2",  xlab="PC1", ylab="PC2",
     xlim=lim, ylim=lim)
text(L[,1], L[, 2], labels=rownames(L), cex=0.8, col="blue", pos=1)
abline(v=0, h=0)
arrows(0,0, L[,1], L[, 2], col=2, code=2, length=0.1)

fml=mlfa$scores
round(fml, 3)
lim<-range(pretty(fml))
plot(fml[,1], fml[,2],main="(a) Factor Scores : PC1 and PC2",  xlab="PC1", ylab="PC2",
     xlim=lim, ylim=lim)
text(fml[,1], fml[,2], labels=rownames(fml), cex=0.8, col="blue", pos=1)
abline(v=0, h=0)

biplot(mlfa$scores[,1:2], mlfa$loadings[,1:2], main="(a) Factor Scores : PC1 and PC2",  xlab="PC1", ylab="PC2")

