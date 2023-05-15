setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.5<-read.table("economicview.txt", header=T)
X<-Data1.3.5[, -1]

# Multiple Scatter Plot
plot(X)

# Star Plot
X<-scale(as.matrix(X[, c(1,2,3,5)]))
rownames(X)<-Data1.3.5[, 1]
stars(X,key.loc=c(8,2), full = FALSE)

# Parallel Coordinate Plot
library(gclus)
parcoordlabel<-function(x, col = 1, lty = 1,  var.label=F,...) 
{
     rx <-lapply(X, range, na.rm = TRUE)
    matplot(1L:ncol(x), t(x), type = "l", col = col, lty = 1:nrow(X), lwd=1.5,
        xlab = "", ylab = "", axes = FALSE, 
           ylim=c(-4, 4), xlim=c(1, nrow(X)), ...)
    axis(1, at = 1L:ncol(x), labels = colnames(x))
    legend("top", horiz=F, legend=colnames(X), lty=1:nrow(X), 
         col=1, cex=0.8, lwd=1.5)
    for (i in 1L:ncol(x))
      invisible()
 }
windows(height=5, width=12)
parcoordlabel(t(X))