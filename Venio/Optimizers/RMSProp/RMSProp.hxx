#pragma once
#include "Optimizer.hxx"

class RMSProp : public Optimizer
{
public:
    explicit RMSProp(Model &network) : Optimizer(network)
    {
        _network = network;
        _epsilon = 1e-8;
        _gamma = 0.9;
        _squared_gradient = new Matrixd[_network.getLayersSize()];
        for(size_t i = 0;i < _network.getLayersSize();++i)
        {
            _squared_gradient[i] = _network.getLayerWeights(i).setZero();
        }
    }
    ~RMSProp() 
    {
        delete[] _squared_gradient;
    }
    void updateWeights(double learning_speed, double epoch) override;

private:
    Matrixd* _squared_gradient;
    double _gamma,_epsilon;
};

