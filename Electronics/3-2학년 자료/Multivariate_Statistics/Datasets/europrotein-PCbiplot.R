setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
data9.3.1<-read.table("newspapers.txt", header=T)
X<-data9.3.1
Y<-X

rownames(Y)<-rownames(X)
colnames(Y)<-colnames(X)

pcasvd<-prcomp(Y, scale=T)
summary(pcasvd)
biplot(pcasvd, pc.biplot=T, xlab="Dim1", ylab="Dim2", 
             main=" PC Biplot : pc.biplot() ", cex=1.0, pch=16)
abline(v=0,h=0)



