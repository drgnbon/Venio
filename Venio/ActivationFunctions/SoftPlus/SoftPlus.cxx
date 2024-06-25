#include "SoftPlus.hxx"

double SoftPlus::toActivateValue(double x)
{
    return log(1.0+exp(x));
}

double SoftPlus::toDerivateValue(double x)
{
    return 1.0/(1.0+exp(-x));
}
