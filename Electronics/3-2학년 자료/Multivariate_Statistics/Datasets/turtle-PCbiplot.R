setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data<-read.table("turtle.txt", header=T)
X<-Data[, -4]
X=log(X)
X
F=X[1:24,];F
M=X[25:48, ];M
round(cov(M), 3)
round(cov(F), 3)
round(cov(X), 3)

round(cor(M), 3)
round(cor(F), 3)
round(cor(X), 3)


# PCA for Male 
pca.M<-princomp(M, cor=FALSE)
summary(pca.M loadings=T) # 설명력, 주성분계수
round(pca.M$scores, 3)  # 주성분점수
screeplot(pca.M, type="lines") # 스크리그림

# 주성분 행렬도
biplot(pca.M, scale=0, xlab="1st PC",ylab="2nd PC",
                main="PC Biplot for Male")   
abline(v=0, h=0)


# PCA for Femle
pca.F<-princomp(F, cor=FALSE)
summary(pca.F loadings=T) # 설명력, 주성분계수
round(pca.F$scores, 3)  # 주성분점수
screeplot(pca.F, type="lines") # 스크리그림

# 주성분 행렬도
biplot(pca.F, scale=0, xlab="1st PC",ylab="2nd PC",
                main="PC Biplot for Female")   
abline(v=0, h=0)


# PCA for Male and Fmale
pca.MF<-princomp(X, cor=FALSE)
summary(pca.MF, loadings=T) # 설명력, 주성분계수
round(pca.MF$scores, 3)  # 주성분점수
screeplot(pca.MF, type="lines") # 스크리그림

# 주성분 행렬도
biplot(pca.MF, scale=0, xlab="1st PC",ylab="2nd PC",
                main="PC Biplot for Male and Fmale")   
abline(v=0, h=0)


