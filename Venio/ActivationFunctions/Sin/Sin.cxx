#include "Sin.hxx"

double Sin::toActivateValue(double x)
{
    return sin(x);
}

double Sin::toDerivateValue(double x)
{
    return cos(x);
}
