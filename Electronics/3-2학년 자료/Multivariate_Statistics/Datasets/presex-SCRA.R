# Simple CRA ca() : Matrix for Two-Way Table
O<-matrix(c(81, 68,  60,  38,
            24, 26,  29,  14,
            18, 41,  74,  42,
            36, 57, 161, 157), byrow=T, nrow=4)
rownames(O)<-c("매우나쁨", "거의나쁨", "조금나쁨", "전혀안나쁨")
colnames(O)<-c("매반", "반대", "찬성", "매찬")
library(ca)
sca=ca(O)
par(pty="s")
plot(sca, main="SCRA package ca : 이원분할표")

# Simple CRA ca() : Text for Two-Way Table
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data8.2.1<-read.table("presex.txt", header=T)
O=Data8.2.1
library(ca)
sca=ca(O)
sca
win.graph()
par(pty="s")
plot(sca, main="SCRA package ca : 이원분할표")
 