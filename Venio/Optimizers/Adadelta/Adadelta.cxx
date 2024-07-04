#include "Adadelta.hxx"

void Adadelta::updateWeights(double learning_speed, double epoch)
{

#ifdef CPU_OPTIMIZATION
    for (int i = 1; i < _network.getLayersSize(); ++i)
    {
        gradient = _network.getLayerWeightsGradient(i);

        _squared_gradient[i] = _gamma * _squared_gradient[i].array() + (1 - _gamma) * gradient.array().square();

        delta = sqrt(_squared_updates[i].array() + _epsilon) / sqrt(_squared_gradient[i].array()+ _epsilon) * gradient.array();

        _squared_updates[i] = _gamma* _squared_updates[i].array() + (1 - _gamma)* delta.array().square();

        _network.setLayerWeights(i, _network.getLayerWeights(i) - learning_speed * delta);



        biasGradient = _network.getLayerBiasGradient(i);

        _squared_bias_gradient[i] = _gamma * _squared_bias_gradient[i].array() + (1 - _gamma) * biasGradient.array().square();

        bias_delta = sqrt(_squared_bias_updates[i].array() + _epsilon) / sqrt(_squared_bias_gradient[i].array() + _epsilon) * biasGradient.array();

        _squared_bias_updates[i] = _gamma * _squared_bias_updates[i].array() + (1 - _gamma) * bias_delta.array().square();

        _network.setLayerBias(i, _network.getLayerBias(i) - learning_speed * bias_delta);


    }
#endif


#ifdef GPU_OPTIMIZATION
    for(int i = 1; i < _network.getLayersSize(); ++i)
    {
        gradient = _network.getLayerWeightsGradient(i);

        _squared_gradient[i] = K::sumAA( K::multAS(_gamma , _squared_gradient[i].array() ) , K::multAS((1 - _gamma) , K::sqrA(gradient.array())  ) );

        delta = K::multAA(K::divAA(K::sqrtA(K::sumAS(_squared_updates[i].array(), _epsilon)) , K::sqrtA(K::sumAS(_squared_gradient[i].array(), _epsilon))) , gradient.array());

        _squared_updates[i] = K::sumAA( K::multAS(_gamma , _squared_updates[i].array() ) , K::multAS((1 - _gamma) , K::sqrA(delta.array()) ) );

        _network.setLayerWeights(i,K::subMM(_network.getLayerWeights(i),K::multMS(learning_speed,delta)));



        biasGradient = _network.getLayerBiasGradient(i);

        _squared_bias_gradient[i] = K::sumAA(K::multAS(_gamma, _squared_bias_gradient[i].array()), K::multAS((1 - _gamma), K::sqrA(biasGradient.array())));

        bias_delta = K::multAA(K::divAA(K::sqrtA(K::sumAS(_squared_bias_updates[i].array(), _epsilon)), K::sqrtA(K::sumAS(_squared_bias_gradient[i].array(), _epsilon))), biasGradient.array());
        
        _squared_bias_updates[i] = K::sumAA(K::multAS(_gamma, _squared_bias_updates[i].array()), K::multAS((1 - _gamma), K::sqrA(bias_delta.array())));

        _network.setLayerBias(i, K::subMM(_network.getLayerBias(i), K::multMS(learning_speed,bias_delta)));


    }         
#endif
}

Adadelta::~Adadelta()
{
    delete[] _squared_bias_gradient;
    delete[] _squared_bias_updates;
    delete[] _squared_gradient;
    delete[] _squared_updates;
}