#include "SiLU.hxx"

double SiLU::toActivateValue(double x)
{
    return x * (1.0 / (1.0 + exp(-x) ));
}

double SiLU::toDerivateValue(double x)
{
    return toActivateValue(x) + (1.0 / (1.0 + exp(-x) )) * (1.0 - toActivateValue(x));
}
