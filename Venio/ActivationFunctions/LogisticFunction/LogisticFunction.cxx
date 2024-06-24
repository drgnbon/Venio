#include "LogisticFunction.hxx"

double toActivateValue(double x)
{
    return 1.f / (1.f + exp(-x));
}
double toDerivateValue(double x)
{
    double activatedValue = toActivateValue(x);
    return activatedValue * (1.f - activatedValue);
}
