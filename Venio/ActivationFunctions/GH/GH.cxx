#include "GH.hxx"

double GH::toActivateValue(double x)
{
    return exp(-(x*x));
}

double GH::toDerivateValue(double x)
{
    return -2.0 * x * exp(-(x*x));
}