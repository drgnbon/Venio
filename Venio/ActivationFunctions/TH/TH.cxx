#include "TH.hxx"

double TH::toActivateValue(double x)
{
    return  ( exp(x) - exp(-x) ) / (exp(x) + exp(-x) )  ;
}

double TH::toDerivateValue(double x)
{
    return 1.0 - toActivateValue(x)*toActivateValue(x);
}
