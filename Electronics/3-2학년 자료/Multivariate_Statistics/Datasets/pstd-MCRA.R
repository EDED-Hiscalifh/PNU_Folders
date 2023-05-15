setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
ptsd<-read.table("ptsd.txt", header=T)
v1 <- as.numeric(ptsd$자아통제력>=3)
v2 <- as.numeric(ptsd$인생문제수>=3)
v3<- as.numeric(ptsd$스트레스수>=3)
v4 <- as.numeric(ptsd$가족결속력>=6)
nptsd <- data.frame(ptsd[,1:3],자아통제력=v1,인생문제수=v2, 스트레스수=v3, 가족결속력=v4)
nptsd<-nptsd[, -1] 
library("ca")
mjca(nptsd)
par(pty="s")
plot(mjca(nptsd), main="MCRA : 스트레스증후군 PTSD")

