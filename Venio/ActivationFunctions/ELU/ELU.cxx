#include "ELU.hxx"

double ELU::toActivateValue(double x)
{
    if (x >= 0)
        return 1;
    return 1.0 * (exp(x) - 1.0);
}

double ELU::toDerivateValue(double x)
{
    if (x >= 0.0)
        return 1.0;
    return toActivateValue(x) + 1.0;
}
