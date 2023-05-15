library(shapes)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
skullc<-read.in("neuralskull_control.txt",8,2)
skullh<-read.in("neuralskull_hydro.txt",8,2)
plotshapes(skullc,skullh, symbol=1, joinline=c(1:8,1))

c<-procGPA(skullc)# Control Group
h<-procGPA(skullh) # Treatment Group
plotshapes(c$mshape, h$mshape, joinline=c(1:8,1))

