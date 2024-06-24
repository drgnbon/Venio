#include "SincFunction.hxx"

double SincFunction::toActivateValue(double x)
{
    if (fabs(x) < 1e-8)
        return 1.0;
    return sin(x) / x;
}

double SincFunction::toDerivateValue(double x)
{
    if (fabs(x) < 1e-8)
        return 0.0;
    return (cos(x) / x) - (sin(x) / (x * x));
}
