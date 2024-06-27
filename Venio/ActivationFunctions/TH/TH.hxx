#pragma once
#include "ActivationFunction.hxx"

class TH : public ActivationFunction
{
public:
    TH() = default;
    virtual ~TH() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};