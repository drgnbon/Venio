#include "RMSProp.hxx"

void RMSProp::updateWeights(double learning_speed, double epoch)
{
    for(int i = 0;i < _network.getLayersSize();++i)
    {
        _squared_gradient[i] = _gamma * _squared_gradient[i].array()  +  (1 - _gamma) * _network.getLayerWeightsGradient(i).array().square();
        _network.setLayerWeights(i,_network.getLayerWeights(i)-Matrixd ((_network.getLayerWeightsGradient(i)*learning_speed).array() / sqrt(_squared_gradient[i].array() +_epsilon)));
    }
}


RMSProp::~RMSProp() {
    delete[] _squared_gradient;
}
