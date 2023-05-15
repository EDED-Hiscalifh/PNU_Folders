library(shapes)
library(MASS)
setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
skullc<-read.in("neuralskull_control.txt",8,2)
skullh<-read.in("neuralskull_hydro.txt",8,2)
plotshapes(skullc,skullh, symbol=1, joinline=c(1:8,1))

c<-procGPA(skullc)# Control Group
h<-procGPA(skullh) # Treatment Group
plotshapes(c$mshape, h$mshape, joinline=c(1:8,1))

#Control group 
plot(c$rotated[,1,], c$rotated[,2,], xlab="Dim1", ylab="Dim2" )
points(c$mshape)
lines(c$mshape[c(1:8,1),])

#Treatment group
plot(h$rotated[,1,], h$rotated[,2,], xlab="Dim1", ylab="Dim2" )
points(h$mshape)
lines(h$mshape[c(1:8,1),])

# Joint Mean Shapes
win.graph()
plot(c$mshape[,1], c$mshape[, 2], xlim=c(-15, 15), ylim=c(-15,15), xlab="Dim1", ylab="Dim2" )
lines(c$mshape[c(1:8,1),], lty=1)
par(new=TRUE)
plot(h$mshape[,1], h$mshape[, 2],xlim=c(-15, 15), ylim=c(-15,15),xlab="Dim1", ylab="Dim2" )
lines(h$mshape[c(1:8,1),], lty=3)
legend("bottomright",c("control","treatment"), lty=c(1, 3))


