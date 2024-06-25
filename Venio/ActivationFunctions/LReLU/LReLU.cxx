#include "LReLU.hxx"

double LReLU::toActivateValue(double x)
{
    if (x > 0.0) return x;
    return 0.01*x;
}

double LReLU::toDerivateValue(double x)
{
    if (x > 0.0)return 1.0;
    return 0.01;
}
