setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data<-read.table("censustract.txt", header=T)
X<-Data
X
round(cor(X), 3)
# PCA based on the SD using princomp( )
pca.R<-princomp(X, cor=T)
summary(pca.R, loadings=T) # 설명력, 주성분계수
round(pca.R$scores, 3)  # 주성분점수
screeplot(pca.R, type="lines") # 스크리그림

# 주성분 행렬도
biplot(pca.R, scale=0, xlab="1st PC",ylab="2nd PC",
                main="PC Biplot for Census Tract Data ")   
abline(v=0, h=0)
