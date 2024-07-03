#include "ADAM.hxx"

ADAM::~ADAM()
{
    delete[] _history_speed_weights;
    delete[] _history_moment_weights;
    delete[] _history_speed_bias;
    delete[] _history_moment_bias;
}

void ADAM::updateWeights(double learning_speed, double epoch)
{
    for (int i = 0; i < _network.getLayersSize(); i++)
    {
        _modified_gamma_epoch = 1 - pow(_gamma, epoch + 1);
        _modified_alfa_epoch = 1 - pow(_alfa, epoch + 1);

        _weights_gradient = _network.getLayerWeightsGradient(i);
        _weights_gradient_squared = _weights_gradient.array().square();

        _bias_gradient = _network.getLayerBiasGradient(i);
        _bias_gradient_squared = _bias_gradient.array().square();

#ifdef CPU_OPTIMIZATION
        _history_speed_weights[i] = _gamma * _history_speed_weights[i] + _modified_gamma * _weights_gradient;

        _history_moment_weights[i] = _alfa * _history_moment_weights[i] + _modified_alfa * _weights_gradient_squared;

        Arrayd _history_speed_weights_corrected = _history_speed_weights[i] / _modified_gamma_epoch;

        Arrayd _history_moment_weights_corrected = (_history_moment_weights[i].array() / _modified_alfa_epoch).sqrt() + _epsilon;

        _network.setLayerWeights(i,_network.getLayerWeights(i) - (_history_speed_weights_corrected / _history_moment_weights_corrected).matrix() * learning_speed);

     
        _history_speed_bias[i] = _gamma * _history_speed_bias[i] + _modified_gamma * _bias_gradient;

        _history_moment_bias[i] = _alfa * _history_moment_bias[i] + _modified_alfa * _bias_gradient_squared;

        Arrayd _history_speed_bias_corrected = _history_speed_bias[i] / _modified_gamma_epoch;

        Arrayd _history_moment_bias_corrected = (_history_moment_bias[i].array() / _modified_alfa_epoch).sqrt() + _epsilon;

        _network.setLayerBias(i, _network.getLayerBias(i) - (_history_speed_bias_corrected / _history_moment_bias_corrected).matrix() * learning_speed);
#endif

#ifdef GPU_OPTIMIZATION
        _history_speed_weights[i] = K::sumMM(K::multMS(_history_speed_weights[i], _gamma), K::multMS(_weights_gradient, _modified_gamma));

        _history_moment_weights[i] = K::sumMM(K::multMS(_history_moment_weights[i], _alfa), K::multMS(_weights_gradient_squared, _modified_alfa));

        Arrayd _history_speed_weights_corrected = K::divMS(_history_speed_weights[i], _modified_gamma_epoch);

        Arrayd _history_moment_weights_corrected = K::sumAS(K::sqrtA(K::divMS(_history_moment_weights[i], _modified_alfa_epoch).array()), _epsilon);

        _network.setLayerWeights(i, K::subMM(_network.getLayerWeights(i),K::multAS(K::divAA(_history_speed_weights_corrected, _history_moment_weights_corrected), learning_speed).matrix()));


        _history_speed_bias[i] = K::sumMM(K::multMS(_history_speed_bias[i], _gamma), K::multMS(_bias_gradient, _modified_gamma));

        _history_moment_bias[i] = K::sumMM(K::multMS(_history_moment_bias[i], _alfa), K::multMS(_bias_gradient_squared, _modified_alfa));

        Arrayd _history_speed_bias_corrected = K::divMS(_history_speed_bias[i], _modified_gamma_epoch);

        Arrayd _history_moment_bias_corrected = K::sumAS(K::sqrtA(K::divMS(_history_moment_bias[i], _modified_alfa_epoch).array()),_epsilon);

        _network.setLayerBias(i, K::subMM(_network.getLayerBias(i), K::multAS(K::divAA(_history_speed_bias_corrected, _history_moment_bias_corrected), learning_speed).matrix()));
#endif
    }


    
}
