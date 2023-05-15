# Simple CRA : smoker data
library(ca)
data(smoke)
O=smoke
sca=ca(O)
sca
win.graph()
par(pty="s")
plot(sca, main="SCRA : 지위와 흡연습관 분할표")