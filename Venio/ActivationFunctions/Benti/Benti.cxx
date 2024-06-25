#include "Benti.hxx"

double Benti::toActivateValue(double x)
{
    return ((sqrt((x * x) + 1.0) - 1.0) / 2.0) + x;
}

double Benti::toDerivateValue(double x)
{
    return 1.0;
}
