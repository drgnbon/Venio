#pragma once

#include "Config.hxx"

class LossFunction
{
public:
    virtual double getMediumLoss(const Matrixd &activeValue, const Matrixd &rightAnswer) = 0;
    virtual Matrixd getDerivationLoss(Matrixd activeValue, Matrixd rightAnswer) = 0;
};
