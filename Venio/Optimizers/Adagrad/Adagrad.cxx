#include "Adagrad.hxx"

void Adagrad::updateWeights(double learning_speed, double epoch)
{

    for(int i = 0;i < _network.getLayersSize();++i)
    {
        _squared_gradient[i] =_squared_gradient[i].array() +  _network.getLayerWeightsGradient(i).array().square();
        _network.setLayerWeights(i,_network.getLayerWeights(i)-Matrixd ((learning_speed * _network.getLayerWeightsGradient(i)).array() / sqrt(_squared_gradient[i].array() + _epsilon).array()));
    }
}

Adagrad::~Adagrad() {
    delete[] _squared_gradient;
}
