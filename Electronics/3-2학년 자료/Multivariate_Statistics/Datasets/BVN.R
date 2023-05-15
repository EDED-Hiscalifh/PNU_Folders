library("mvtnorm")
x1 <- x2 <- seq(-10, 10, length = 51)
dens <- matrix(dmvnorm(expand.grid(x1, x2),
sigma = rbind(c(1, 0.5), c(1, 0.5))),
ncol = length(x1))
s3d <- scatterplot3d(x1, x2,
seq(min(dens), max(dens), length = length(x1)),
type = "n", grid = FALSE, angle = 70,
zlab = expression(f(x[1], x[2])),
xlab = expression(x[1]), ylab = expression(x[2]),
main = "Bivariate normal distribution")
text(s3d$xyz.convert(-1, 10, 0.07),
labels = expression(f(x) == frac(1, sqrt((2 * pi)^n *
phantom(".") * det(Sigma[X]))) * phantom(".") * exp * {
bgroup("(", - scriptstyle(frac(1, 2) * phantom(".")) *
(x - mu)^T * Sigma[X]^-1 * (x - mu), ")")}))
text(s3d$xyz.convert(1.5, 10, 0.05),
labels = expression("with" * phantom("m") *
mu == bgroup("(", atop(0, 0), ")") * phantom(".") * "," *
phantom(0) *
{Sigma[X] == bgroup("(", atop(1 * phantom(0) * 0.5,
0.5 * phantom(0) * 1), ")")}))
for(i in length(x1):1)
s3d$points3d(rep(x1[i], length(x2)), x2, dens[i,], type = "l")
for(i in length(x2):1)
s3d$points3d(x1, rep(x2[i], length(x1)), dens[,i], type = "l")