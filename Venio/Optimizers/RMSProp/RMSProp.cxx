#include "RMSProp.hxx"

void RMSProp::updateWeights(double learning_speed, double epoch)
{
    for (size_t i = 0; i < _network.getLayersSize(); ++i)
    {
        _weights_gradient = _network.getLayerWeightsGradient(i);
        _bias_gradient = _network.getLayerBiasGradient(i);

#ifdef CPU_OPTIMIZATION
        _squared_gradient_weights[i] = _gamma * _squared_gradient_weights[i].array() + _modified_gamma * _weights_gradient.array().square();

        Matrixd _squared_gradient_weights_corrected = (_weights_gradient * learning_speed).array() / (sqrt(_squared_gradient_weights[i].array() + _epsilon));

        _network.setLayerWeights(i, _network.getLayerWeights(i) - _squared_gradient_weights_corrected);


        _squared_gradient_bias[i] = _gamma * _squared_gradient_bias[i].array() + _modified_gamma * _bias_gradient.array().square();

        Matrixd _squared_gradient_bias_corrected = (_bias_gradient * learning_speed).array() / (sqrt(_squared_gradient_bias[i].array() + _epsilon));

        _network.setLayerBias(i, _network.getLayerBias(i) - _squared_gradient_bias_corrected);
#endif


#ifdef GPU_OPTIMIZATION
        _squared_gradient_weights[i] = K::sum(K::scalarMultiply(_squared_gradient_weights[i].array(), _gamma), K::scalarMultiply(_weights_gradient.array().square(), _modified_gamma));

        Matrixd _squared_gradient_weights_corrected = K::divideArrays(K::scalarMultiply(_weights_gradient, learning_speed).array(), sqrt(K::scalarSum(_squared_gradient_weights[i].array(), _epsilon)));

        _network.setLayerWeights(i, K::sub(_network.getLayerWeights(i), _squared_gradient_weights_corrected));


        _squared_gradient_bias[i] = K::sum(K::scalarMultiply(_squared_gradient_bias[i].array(), _gamma), K::scalarMultiply(_bias_gradient.array().square(), _modified_gamma));

        Matrixd _squared_gradient_bias_corrected = K::divideArrays(K::scalarMultiply(_bias_gradient, learning_speed).array(), sqrt(K::scalarSum(_squared_gradient_bias[i].array(), _epsilon)));

        _network.setLayerBias(i, K::sub(_network.getLayerBias(i), _squared_gradient_bias_corrected));
#endif
    }
}
RMSProp::~RMSProp()
{
    delete[] _squared_gradient_weights;
    delete[] _squared_gradient_bias;
}