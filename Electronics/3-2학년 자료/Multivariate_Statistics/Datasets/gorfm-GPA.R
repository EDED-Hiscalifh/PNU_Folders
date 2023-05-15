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
plotshapes(gorft, gormt, symbol=1,joinline=c(1,5,4,3,2,8,7,6,1))

gorf<-procGPA(gorft)
gorm<-procGPA(gormt)
plotshapes(gorf$mshape, gorm$mshape, symbol=1, joinline=c(1,5,4,3,2,8,7,6,1))

