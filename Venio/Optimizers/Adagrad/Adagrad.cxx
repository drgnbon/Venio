#include "Adagrad.hxx"

void Adagrad::updateWeights(double learning_speed, double epoch)
{

#ifdef CPU_OPTIMIZATION

    for(int i = 0;i < _network.getLayersSize();++i)
    {

        gradient = _network.getLayerWeightsGradient(i);
        _squared_gradient[i] =_squared_gradient[i].array() + gradient.array().square();
        _network.setLayerWeights(i,_network.getLayerWeights(i)-Matrixd ((learning_speed * gradient).array() / sqrt(_squared_gradient[i].array() + _epsilon).array()));
        
        bias_gradient = _network.getLayerBiasGradient(i);
        _squared_bias_gradient[i] = _squared_bias_gradient[i].array() + bias_gradient.array().square();
        _network.setLayerBias(i, _network.getLayerBias(i) - Matrixd((learning_speed * bias_gradient).array() / sqrt(_squared_bias_gradient[i].array() + _epsilon).array()));

    }

#endif

#ifdef GPU_OPTIMIZATION

    for (int i = 0; i < _network.getLayersSize(); ++i)
    {

        gradient = _network.getLayerWeightsGradient(i);

        _squared_gradient[i] += Matrixd(gradient.array().square());
        _network.setLayerWeights(i, K::sub(_network.getLayerWeights(i), K::scalarMultiply(learning_speed, Matrixd(K::divideArrays(gradient.array(), K::sqrt(K::scalarAdd(_squared_gradient[i].array(), _epsilon).array()))))));

        bias_gradient = _network.getLayerBiasGradient(i);

        _squared_bias_gradient[i] += Matrixd(bias_gradient.array().square());
        _network.setLayerBias(i, K::sub(_network.getLayerBias(i), K::scalarMultiply(learning_speed, Matrixd(K::divideArrays(bias_gradient.array(), K::sqrt(K::scalarAdd(_squared_bias_gradient[i].array(), _epsilon).array()))))));

    }
#endif

    


}

Adagrad::~Adagrad() {
    delete[] _squared_bias_gradient;
    delete[] _squared_gradient;
}
