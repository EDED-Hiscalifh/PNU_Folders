setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
view<-read.table("view.txt", header=T)
view<-view[, -1]
##Multiple CA Plot
library("ca")
mjca(view)
par(pty="s")
plot(mjca(view), main="MCRA : 성별, 나이 수입에 따른 경제전망과 정책선호도")
