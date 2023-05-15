setwd("c:/R과 함께하는 다변량자료분석/R_code_data")
Data1.3.6<-read.table("sucidefreq.txt")
Data1.3.6
freq<-as.matrix(Data1.3.6)

sex<-c("Male", "Female")
age<- c("10-20", "25-35", "40-50", "55-65", "70-90")
method<-c("Poison", "Gas",  "Hang", "Drown", "Gun", "Jump")
data<-expand.grid(Sex=sex, Age=age, Method=method)
data<-cbind(data, freq)

# Makeing a 3-way table
table<-xtabs(freq~Sex+Age+Method, data=data)

# Print table
ftable(table)

# Chi-square test of indepedence
summary(table)

# Mosaic Plot
mosaicplot(table, main="Mosaic Plot for Sucide Rates")

