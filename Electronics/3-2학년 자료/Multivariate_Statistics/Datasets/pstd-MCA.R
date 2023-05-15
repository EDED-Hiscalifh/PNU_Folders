setwd("C:/Users/stat/Desktop/개인_WORK/외부초청강의/부산발전연구원")
ptsd<-read.table("ptsd.txt", header=T)
ptsd
library("ca")

v1 <- as.numeric(ptsd$자아통제력>=3)
v2 <- as.numeric(ptsd$인생문제수>=3)
v3<- as.numeric(ptsd$스트레스수>=3)
v4 <- as.numeric(ptsd$가족결속력>=6)

nptsd <- data.frame(ptsd[,1:3],자아통제력=v1,인생문제수=v2, 스트레스수=v3, 가족결속력=v4)

nptsd<-nptsd[, -1] 

mjca(nptsd)
par(pty="s")
plot(mjca(nptsd), main="Multiple CA for PTSD")


