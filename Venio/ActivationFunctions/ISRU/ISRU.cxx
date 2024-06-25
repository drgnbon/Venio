#include "ISRU.hxx"

double ISRU::toActivateValue(double x)
{
    return x/ sqrt(1.0+1.0*pow(x,2.0));
}

double ISRU::toDerivateValue(double x)
{
    return pow(1.0/ sqrt(1.0+1.0*pow(x,2.0)),3.0);
}

