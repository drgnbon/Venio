#pragma once
#include "ActivationFunction.hxx"

class SoftPlus : public ActivationFunction
{
public:
    SoftPlus() = default;
    virtual ~SoftPlus() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};