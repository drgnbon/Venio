#pragma once
#include "ActivationFunction.hxx"

class SincFunction : public ActivationFunction
{
public:
    SincFunction() = default;
    virtual ~SincFunction() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};