#pragma once
#include "ActivationFunction.hxx"

class LinearFunction : public ActivationFunction
{
public:
    LinearFunction() = default;
    virtual ~LinearFunction() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};