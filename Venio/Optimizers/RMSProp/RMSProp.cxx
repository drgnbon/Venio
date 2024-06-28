#include "RMSProp.hxx"

void RMSProp::updateWeights(double learning_speed, double epoch)
{
    for (size_t i = 0; i < _network.getLayersSize(); ++i)
    {
        _weights_gradient = _network.getLayerWeightsGradient(i);

#ifdef CPU_OPTIMIZATION
        _squared_gradient[i] = _gamma * _squared_gradient[i].array() + _modified_gamma * _weights_gradient.array().square();

        Matrixd _squared_gradient_corrected = (_weights_gradient * learning_speed).array() / (sqrt(_squared_gradient[i].array() + _epsilon));

        _network.setLayerWeights(i, _network.getLayerWeights(i) - _squared_gradient_corrected);
#endif

#ifdef GPU_OPTIMIZATION
        _squared_gradient[i] = K::sum(K::scalarMultiply(_squared_gradient[i].array(), _gamma), K::scalarMultiply(_weights_gradient.array().square(), _modified_gamma));

        Matrixd _squared_gradient_corrected = K::divideArrays(K::scalarMultiply(_weights_gradient, learning_speed).array(), sqrt(K::scalarSum(_squared_gradient[i].array(), _epsilon)));

        _network.setLayerWeights(i, K::sub(_network.getLayerWeights(i), _squared_gradient_corrected));
#endif
    }
}
RMSProp::~RMSProp()
{
    delete[] _squared_gradient;
}