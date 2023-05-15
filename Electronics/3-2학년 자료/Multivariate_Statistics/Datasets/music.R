music = c(210, 194, 170, 110,
190, 406, 730, 290)
dim(music) = c(2, 2, 2)
dimnames(music) =
list(Age = c("Old", "Young"),
Education = c("High", "Low"),
Listen = c("Yes", "No"))
mosaicplot(music)
mosaicplot(music, col = hcl(240),
main = "Classical Music Listening")