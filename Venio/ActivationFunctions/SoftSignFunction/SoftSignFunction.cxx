#include "SoftSignFunction.hxx"

double SoftSignFunction::toActivateValue(double x)
{
    return x / (1.0 + fabs(x));
}

double SoftSignFunction::toDerivateValue(double x)
{
    return 1.0 / std::pow(1.0 + fabs(x), 2);
}
