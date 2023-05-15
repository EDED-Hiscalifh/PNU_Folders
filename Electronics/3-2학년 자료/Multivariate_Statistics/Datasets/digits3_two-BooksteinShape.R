library(shapes)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
digit3<-read.in("digit3.txt",13,2)
digit3_two<-digit3[,, 2:3]
digit3_two
bookdigit3<-bookstein2d(digit3_two)
bookdigit3
plotshapes(digit3_two, symbol=1, joinline=c(1:13))
plotshapes(bookdigit3$bshpv,color=1:6, symbol=1,  joinline=c(1:13))
plotshapes(bookdigit3$mshape,color=1:6, symbol=1,  joinline=c(1:13))
