#pragma once
#include "ActivationFunction.hxx"

class ISRU : public ActivationFunction
{
public:
    ISRU() = default;
    virtual ~ISRU() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};