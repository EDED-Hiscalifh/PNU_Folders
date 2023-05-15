setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
XX<-read.table("irisflower.txt",header=T)
XX<-XX[,2:5]
XX<-as.matrix(XX)

SandR<-array(NA,c(4,4,6))
rownames(SandR)<-colnames(XX)
colnames(SandR)<-colnames(XX)
variation<-matrix(NA,3,4)
rownames(variation)<-c("setosa","versicolor","virginica")
colnames(variation)<-c("detS","trS", "detR","trR")
for(i in 1:3) {
X<-XX[(50*(i-1)+1):(50*i),]

S<-cov(X)
R<-cor(X)

detS<-det(S)
detR<-det(R)
trS<-sum(diag(S))
trR<-sum(diag(R))

SandR[,,i*2-1]<-S
SandR[,,i*2]<-R

variation[i,1]<-detS
variation[i,2]<-trS
variation[i,3]<-detR
variation[i,4]<-trR
}

SandR<-round(SandR,3)
variation<-round(variation,6)
setosa<-list(SandR[,,1],SandR[,,2],variation[1,])
versicolor<-list(SandR[,,3],SandR[,,4],variation[2,])
virginica<-list(SandR[,,5],SandR[,,6],variation[3,])

setosa
versicolor
virginica