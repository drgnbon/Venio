#include "RMSProp.hxx"

void RMSProp::updateWeights(double learning_speed, double epoch)
{
    for(size_t i = 0;i < _network.getLayersSize();++i)
    {
        _squared_gradient[i] = K::sum(K::scalarMultiply(_squared_gradient[i].array(), _gamma), K::scalarMultiply(_network.getLayerWeightsGradient(i).array().square(), (1 - _gamma)));
        _network.setLayerWeights(i, K::sub(_network.getLayerWeights(i), Matrixd(K::divideArrays(K::scalarMultiply(_network.getLayerWeightsGradient(i), learning_speed).array(), sqrt(K::scalarSum(_squared_gradient[i].array(), _epsilon))))));


        //_squared_gradient[i] = _gamma * _squared_gradient[i].array()  +  (1 - _gamma) * _network.getLayerWeightsGradient(i).array().square();
        //_network.setLayerWeights(i,_network.getLayerWeights(i)-Matrixd ((_network.getLayerWeightsGradient(i)*learning_speed).array() / sqrt(_squared_gradient[i].array() +_epsilon)));
    }
}