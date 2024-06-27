#include "GD.hxx"



void GD::updateWeights(double learning_speed, double epoch)
{
    for (int i = 1; i < _network.getLayersSize(); i++)
    {
        _network.setLayerWeights(i, K::sub(_network.getLayerWeights(i), K::scalarMultiply(_network.getLayerWeightsGradient(i), learning_speed)));
        _network.setLayerBias(i, K::sub(_network.getLayerBias(i), K::scalarMultiply(_network.getLayerBiasGradient(i), learning_speed)));

        /*_network.setLayerWeights(i, _network.getLayerWeights(i) - _network.getLayerWeightsGradient(i) * learning_speed);
        _network.setLayerBias(i, _network.getLayerBias(i) - _network.getLayerBiasGradient(i) * learning_speed);*/
    }
}
