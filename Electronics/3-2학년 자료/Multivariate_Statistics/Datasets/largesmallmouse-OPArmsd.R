library(shapes)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
mouseT2<-read.in("mouseT2.txt",6,2)
cm<-mouseT2[,,1:30] # 대조그룹 n1=30
sm<-mouseT2[,,31:53] # 작은그룹 n2=23
lm<-mouseT2[,,54:76] # 큰 그룹 n3=23

sm1<-sm[,,1]
lm1<-lm[,,1]
plotshapes(lm1, sm1, symbol=1, joinline=c(1, 6, 2, 3, 4, 5,1))
PA1<-procOPA(lm1, sm1) #matching sm1 onto lm1
plotshapes(PA1$Bhat, symbol=1, joinline=c(1, 6, 2, 3, 4, 5,1))
PA2<-procOPA(sm1, lm1) #matching lm1 onto sm1
plotshapes(PA2$Bhat, symbol=1, joinline=c(1, 6, 2, 3, 4, 5,1))

#Shape Vatiability
OPSS_AB=PA1$OSS
RMSD_AB = PA1$rmsd
list(OPSS_AB, RMSD_AB)

OPSS_BA=PA2$OSS
RMSD_BA = PA2$rmsd
list(OPSS_BA, RMSD_BA)