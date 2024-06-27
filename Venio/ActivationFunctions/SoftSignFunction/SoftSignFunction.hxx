#pragma once
#include "ActivationFunction.hxx"

class SoftSignFunction : public ActivationFunction{
public:
    SoftSignFunction() = default;
    virtual ~SoftSignFunction() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};