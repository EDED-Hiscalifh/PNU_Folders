#include <stdio.h>
#include <math.h>
#define g 9.8
void main()
{
	double m, r, h, t, i;

	printf("ют╥б:");
	scanf("%lf %lf %lf %lf", &m,&r,&h,&t);
	i = m*r*r*(g*t*t / 2 * h - 1);
	printf("%.12f", i);
}