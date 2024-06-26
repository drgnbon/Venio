#pragma once
#include "Optimizer.hxx"

class GD : public Optimizer
{
public:
    explicit GD(Model &network) : Optimizer(network) {}

    void updateWeights(double learning_speed, double epoch) override;
};