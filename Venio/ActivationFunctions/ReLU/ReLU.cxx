#include "ReLU.hxx"

double ReLU::toActivateValue(double x)
{
    if (x > 0.0) return x;
    return 0.0;
}

double ReLU::toDerivateValue(double x)
{
    if (x > 0.0)return 1.0;
    return 0.0;
}
