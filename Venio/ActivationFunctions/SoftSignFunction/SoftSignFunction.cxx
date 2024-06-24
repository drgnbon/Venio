#include "SoftSignFunction.hxx"

double toActivateValue(double x)
{
    return x / (1.0 + fabs(x));
}

double toDerivateValue(double x)
{
    return 1.0 / std::pow(1.0 + fabs(x), 2);
}
