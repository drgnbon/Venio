#pragma once
#include "ActivationFunction.hxx"

class Benti : public ActivationFunction
{
public:
    Benti() = default;
    virtual ~Benti() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};