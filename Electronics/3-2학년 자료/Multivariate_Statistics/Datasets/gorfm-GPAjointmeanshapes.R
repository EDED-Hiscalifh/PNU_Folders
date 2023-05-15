library(shapes)
# female and male Gorillas (Dryden and Mardia, 1998)
data(gorf.dat)
data(gorm.dat)
n1<-dim(gorf.dat)[3]
n2<-dim(gorm.dat)[3]

gorft<-array(0,dim=c(8,2, n1))
for(i in 1:n1){
T<-matrix(c(cos(90),sin(90),
           -sin(90), cos(90)), byrow=T, nrow=2)
gorft[,,i]<-gorf.dat[,,i]%*%T
}
gormt<-array(0,dim=c(8,2, n2))
for(i in 1:n2){
T<-matrix(c(cos(90), sin(90),
           -sin(90), cos(90)), byrow=T, nrow=2)
gormt[,,i]<-gorm.dat[,,i]%*%T
}

join<-c(1,5,4,3,2,8,7,6,1)
plotshapes(gorft, gormt, symbol=1,joinline=join)

gorf<-procGPA(gorft)
gorm<-procGPA(gormt)
plotshapes(gorf$mshape, gorm$mshape, symbol=1, joinline=join)

#Female group 
plot(gorf$rotated[,1,], gorf$rotated[,2,], xlab="Dim1", ylab="Dim2" )
points(gorf$mshape)
lines(gorf$mshape[join,])

#Male group
plot(gorm$rotated[,1,],gorm$rotated[,2,], xlab="Dim1", ylab="Dim2" )
points(gorm$mshape)
lines(gorm$mshape[join,])

# Joint Mean Shapes
plot(gorf$mshape[,1], gorf$mshape[, 2],xlim=c(-150, 150), ylim=c(-100,80), xlab="Dim1", ylab="Dim2" )
lines(gorf$mshape[join,], lty=1)
par(new=TRUE)
plot(gorm$mshape[,1], gorm$mshape[, 2],xlim=c(-150, 150), ylim=c(-100,80), xlab="Dim1", ylab="Dim2" )
lines(gorm$mshape[join,], lty=3)
legend("bottomright",c("female","male"), lty=c(1, 3))