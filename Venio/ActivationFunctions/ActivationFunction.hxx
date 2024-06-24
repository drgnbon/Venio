#pragma once

#include "Config.hxx"

class ActivationFunction
{
public:
    ActivationFunction() = default;
    ~ActivationFunction() = default;
    virtual double toActivateValue(double x) = 0;
    virtual double toDerivateValue(double x) = 0;

    Matrixd toActivateMatrix(Matrixd matrix);
    Matrixd toDerivateMatrix(Matrixd matrix);
};
