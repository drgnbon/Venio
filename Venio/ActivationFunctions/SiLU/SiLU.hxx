#pragma once
#include "ActivationFunction.hxx"

class SiLU : public ActivationFunction
{
public:
    SiLU() = default;
    virtual ~SiLU() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};