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
        _squared_gradient_weights[i] = K::sumAA(K::multAS(_squared_gradient_weights[i].array(), _gamma), K::multAS(K::sqrA(_weights_gradient.array()), _modified_gamma));

        Matrixd _squared_gradient_weights_corrected = K::divAA(K::multMS(_weights_gradient, learning_speed).array(), sqrt(K::sumAS(_squared_gradient_weights[i].array(), _epsilon)));

        _network.setLayerWeights(i, K::subMM(_network.getLayerWeights(i), _squared_gradient_weights_corrected));


        _squared_gradient_bias[i] = K::sumAA(K::multAS(_squared_gradient_bias[i].array(), _gamma), K::multAS(K::sqrA(_bias_gradient.array()), _modified_gamma));

        Matrixd _squared_gradient_bias_corrected = K::divAA(K::multMS(_bias_gradient, learning_speed).array(), sqrt(K::sumAS(_squared_gradient_bias[i].array(), _epsilon)));

        _network.setLayerBias(i, K::subMM(_network.getLayerBias(i), _squared_gradient_bias_corrected));
#endif
    }
}
RMSProp::~RMSProp()
{
    delete[] _squared_gradient_weights;
    delete[] _squared_gradient_bias;
}