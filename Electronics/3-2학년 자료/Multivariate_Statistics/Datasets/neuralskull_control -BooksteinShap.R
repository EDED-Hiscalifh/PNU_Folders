library(shapes)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
nsc<-read.in("neuralskull_control.txt",8,2)
booknsc<-bookstein2d(nsc, l1=1, l2=8)
booknsc
plotshapes(nsc, symbol=1, joinline=c(1:8, 1))
plotshapes(booknsc$bshpv,symbol=1,  joinline=c(1:8, 1))
plotshapes(booknsc$mshape, symbol=1,  joinline=c(1:8, 1))
