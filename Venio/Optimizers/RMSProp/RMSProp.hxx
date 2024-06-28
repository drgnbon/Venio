#pragma once
#include "Optimizer.hxx"

class RMSProp : public Optimizer
{
public:
    explicit RMSProp(Model& network) : Optimizer(network)
    {
        _network = network;
        size_t layer_count = _network.getLayersSize();

        _epsilon = 1e-8;
        _gamma = 0.9;
        _modified_gamma = 1 - _gamma;
        _squared_gradient = new Matrixd[layer_count];

        for (size_t i = 0; i < layer_count; ++i)
        {
            _squared_gradient[i] = _network.getLayerWeights(i).setZero();
        }
    }
    ~RMSProp();
    void updateWeights(double learning_speed, double epoch) override;

private:
    Matrixd* _squared_gradient;
    Matrixd _weights_gradient;
    double _gamma, _epsilon;
    double _modified_gamma;
};

