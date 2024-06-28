#include "ADAM.hxx"


void ADAM::updateWeights(double learning_speed, double epoch)
{

    for (int i = 0; i < _network.getLayersSize(); i++)
    {
        double modified_gamma = 1 - _gamma;
        double modified_alfa = 1 - _alfa;
        double modified_gamma_epoch = 1 - pow(_gamma, epoch + 1);
        double modified_alfa_epoch = 1 - pow(_alfa, epoch + 1);

        Matrixd weights_gradient = _network.getLayerWeightsGradient(i);
        Matrixd weights_gradient_squared = weights_gradient.array().square();

        Matrixd bias_gradient = _network.getLayerBiasGradient(i);
        Matrixd bias_gradient_squared = bias_gradient.array().square();



#ifdef CPU_OPTIMIZATION
        _history_speed_weights[i] = _gamma * _history_speed_weights[i] + 
            modified_gamma * weights_gradient;

        _history_moment_weights[i] = _alfa * _history_moment_weights[i] +
            modified_alfa * weights_gradient_squared;

        Arrayd _history_speed_weights_corrected = _history_speed_weights[i] / modified_gamma_epoch;

        Arrayd _history_moment_weights_corrected = (_history_moment_weights[i].array() / modified_alfa_epoch).sqrt() + _epsilon;


        _network.setLayerWeights(i,_network.getLayerWeights(i) - (_history_speed_weights_corrected / _history_moment_weights_corrected).matrix() * learning_speed);

     
        _history_speed_bias[i] = _gamma * _history_speed_bias[i] +
            modified_gamma * bias_gradient;

        _history_moment_bias[i] = _alfa * _history_moment_bias[i] +
            modified_alfa * bias_gradient_squared;

        Arrayd _history_speed_bias_corrected = _history_speed_bias[i] / modified_gamma_epoch;

        Arrayd _history_moment_bias_corrected = (_history_moment_bias[i].array() / modified_alfa_epoch).sqrt() + _epsilon;

        _network.setLayerBias(i, _network.getLayerBias(i) - (_history_speed_bias_corrected / _history_moment_bias_corrected).matrix() * learning_speed);
#endif



        _history_speed_weights[i] = K::sum(K::scalarMultiply(_history_speed_weights[i], _gamma), K::scalarMultiply(weights_gradient, modified_gamma));

        _history_moment_weights[i] = K::sum(K::scalarMultiply(_history_moment_weights[i], _alfa), K::scalarMultiply(weights_gradient_squared, modified_alfa));

        Arrayd _history_speed_weights_corrected = K::scalarDivide(_history_speed_weights[i], modified_gamma_epoch);

        Arrayd _history_moment_weights_corrected = K::scalarDivide(_history_moment_weights[i], modified_alfa_epoch).array().sqrt() + _epsilon;

        _network.setLayerWeights(i, K::sub(_network.getLayerWeights(i),K::scalarMultiply(K::divideArrays(_history_speed_weights_corrected, _history_moment_weights_corrected), learning_speed)));



        _history_speed_bias[i] = K::sum(K::scalarMultiply(_history_speed_bias[i], _gamma), K::scalarMultiply(bias_gradient, modified_gamma));

        _history_moment_bias[i] = K::sum(K::scalarMultiply(_history_moment_bias[i], _alfa), K::scalarMultiply(bias_gradient_squared, modified_alfa));

        Arrayd _history_speed_bias_corrected = K::scalarDivide(_history_speed_bias[i], modified_gamma_epoch);

        Arrayd _history_moment_bias_corrected = K::scalarDivide(_history_moment_bias[i], modified_alfa_epoch).array().sqrt() + _epsilon;

        _network.setLayerBias(i, K::sub(_network.getLayerBias(i), K::scalarMultiply(K::divideArrays(_history_speed_bias_corrected, _history_moment_bias_corrected), learning_speed)));


    }
}
