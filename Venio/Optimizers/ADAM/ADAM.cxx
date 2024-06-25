#include "ADAM.hxx"


void ADAM::updateWeights(double learning_speed, double epoch)
{

    for (int i = 0; i < _network.getLayersSize(); i++)
    {
        _history_speed[i] = (_gamma * _history_speed[i]) + ((1 - _gamma) * _network.getLayerWeightsGradient(i));
        _history_moment[i] = (_alfa * _history_moment[i]) + Eigen::MatrixXd((1 - _alfa) * _network.getLayerWeightsGradient(i).array().pow(2));
        _network.setLayerWeights(i, _network.getLayerWeights(i) - learning_speed * Eigen::MatrixXd((_history_speed[i] / (1 - pow(_gamma, epoch + 1))).array().cwiseQuotient((_history_moment[i] / (1 - pow(_alfa, epoch + 1))).array().sqrt() + _epsilon).array()));
    }
}
