setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.2<-read.table("klpga.txt", header=T)
X<-Data1.3.2

# PCA based on the SD using princomp( )
pca.R<-princomp(X, cor=T)
summary(pca.R, loadings=T) # 설명력, 주성분계수
round(pca.R$scores, 3)  # 주성분점수
screeplot(pca.R, type="lines") # 스크리그림

# 주성분 행렬도
biplot(pca.R, scale=0, xlab="1st PC",ylab="2nd PC",
                main="PC Biplot for KLPGA Data ")   
abline(v=0, h=0)


# PCA on the SVD using prcomp( )
pcasvd.Z<-prcomp(X, scale=T) 
summary(pcasvd.Z)  # 설명력
round(pcasvd.Z$rotation, 3) # 주성분계수
pcasvd.Z$scale
screeplot(pcasvd.Z, type="lines") #스크리그림

# 주성분 행렬도
biplot(pcasvd.Z, scale=0,  xlab="1st PC",ylab="2nd PC",
                main="PC Biplot for KLPGA Data ")
abline(v=0, h=0)