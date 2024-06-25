#include "Adadelta.hxx"

void Adadelta::updateWeights(double learning_speed, double epoch)
{

    for(int i = 0; i < _network.getLayersSize(); ++i)
    {
        _squared_gradient[i] = _gamma * _squared_gradient[i].array() + (1 - _gamma) * _network.getLayerWeightsGradient(i).array().square();

        Matrixd delta = ((_squared_updates[i].array() + _epsilon).sqrt() / (_squared_gradient[i].array() + _epsilon).sqrt()) * _network.getLayerWeightsGradient(i).array();

        _squared_updates[i] = _gamma * _squared_updates[i].array() + (1 - _gamma) * delta.array().square();

        _network.setLayerWeights(i,_network.getLayerWeights(i)-delta);

    }                                                                                       
}

Adadelta::~Adadelta()
{
    delete[] _squared_gradient;
    delete[] _squared_updates;
}