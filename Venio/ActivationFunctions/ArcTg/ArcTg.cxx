#include "ArcTg.hxx"

double ArcTg::toActivateValue(double x)
{
    return atan(x);
}

double ArcTg::toDerivateValue(double x)
{
    return 1/((x*x)+1);
}
