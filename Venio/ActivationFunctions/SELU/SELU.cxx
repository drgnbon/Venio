#include "SELU.hxx"

double SELU::toActivateValue(double x)
{
    if (x >= 0.0)
        return x * 1.0507;
    return (1.67326 * (exp(x) - 1.0)) * 1.0507;
}

double SELU::toDerivateValue(double x)
{
    if (x >= 0.0)
        return 1.0507;
    return (1.67326 * exp(x)) * 1.0507;
}
