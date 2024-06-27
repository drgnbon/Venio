#pragma once
#include "ActivationFunction.hxx"

class ELU : public ActivationFunction
{
public:
    ELU() = default;
    virtual ~ELU() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};