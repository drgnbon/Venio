#include "ISRLU.hxx"

double ISRLU::toActivateValue(double x)
{
    if(x >= 0)
        return  x;
    return x / sqrt(1.0+1.0*(x*x));
}

double ISRLU::toDerivateValue(double x)
{
    if(x >= 0.0)
        return 1.0;
    return pow((1.0/(sqrt(1.0+1.0*pow(x,2.0)))),3.0);
}
