setwd("C:/Users/stat/Desktop/개인_WORK/외부초청강의/동의대간호학과")
wbc<-read.table("wbc.txt", header=T)
rownames(wbc)<-wbc[,1]
colnames(wbc)
wbc<-wbc[,-1]
library("ca")
ca(wbc)
par(pty="s")
plot(ca(wbc), main="Simple CA for Women Breast Cancer")

