#pragma once
#include "ActivationFunction.hxx"

class ReLU : public ActivationFunction
{
public:
    ReLU() = default;
    virtual ~ReLU() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};