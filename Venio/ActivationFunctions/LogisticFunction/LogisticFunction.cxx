#include "LogisticFunction.hxx"
#include "Kernel.hxx"

double LogisticFunction::toActivateValue(double x)
{
    return 1.f / (1.f + exp(-x));
}
double LogisticFunction::toDerivateValue(double x)
{
    double activatedValue = toActivateValue(x);
    return activatedValue * (1.f - activatedValue);
}
