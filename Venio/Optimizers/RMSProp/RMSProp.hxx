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
        _squared_gradient_weights = new Matrixd[layer_count];
        _squared_gradient_bias = new Matrixd[layer_count];

        for (size_t i = 0; i < layer_count; ++i)
        {
            _squared_gradient_weights[i] = _network.getLayerWeights(i).setZero();
            _squared_gradient_bias[i] = _network.getLayerBias(i).setZero();
        }
    }
    ~RMSProp();
    void updateWeights(double learning_speed, double epoch) override;

private:
    Matrixd* _squared_gradient_weights;
    Matrixd _weights_gradient;
    Matrixd* _squared_gradient_bias;
    Matrixd _bias_gradient;
    double _gamma, _epsilon;
    double _modified_gamma;
};

