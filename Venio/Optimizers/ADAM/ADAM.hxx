// #include "../Optimizer/Optimizer.hxx"

// class ADAM : public Optimizer
// {
// public:
//     Matrixd *_history_speed;
//     Matrixd *_history_moment;
//     double _gamma = 0.9;
//     double _alfa = 0.999;
//     double _epsilon = 1e-8;

//     ADAM(Model &network) : Optimizer(network);
//     ~ADAM();

//     void updateWeights(double learning_speed = 0.5, double epoch = 1);
// };
// class ADAM : public Optimizer
// {
// public:
//     Matrixd* _history_speed;
//     Matrixd* _history_moment;
//     double _gamma = 0.9;
//     double _alfa = 0.999;
//     double _epsilon = 1e-8;

//     ADAM(Model& network) : Optimizer(network)
//     {
//         _network = network;
//         _history_speed = new Matrixd[network.getLayersSize()];
//         _history_moment = new Matrixd[network.getLayersSize()];

//         for (int i = 1; i < network.getLayersSize(); ++i)
//         {
//             _history_speed[i] = network.getLayerWeights(i);
//             _history_moment[i] = network.getLayerWeights(i);
//             _history_speed[i].setZero();
//             _history_moment[i].setZero();
//         }
//     }

//     ~ADAM()
//     {
//         delete[] _history_speed;
//         delete[] _history_moment;
//     }

//     void updateWeights(double learning_speed = 0.5, double epoch = 1)
//     {

//         for (int i = 0; i < _network.getLayersSize(); i++)
//         {
//             _history_speed[i] = (_gamma * _history_speed[i]) + ((1 - _gamma) * _network.getLayerWeightsGradient(i));
//             _history_moment[i] = (_alfa * _history_moment[i]) + Eigen::MatrixXd((1 - _alfa) * _network.getLayerWeightsGradient(i).array().pow(2));
//             _network.setLayerWeights(i, _network.getLayerWeights(i) - learning_speed * Eigen::MatrixXd((_history_speed[i] / (1 - pow(_gamma, epoch + 1))).array().cwiseQuotient((_history_moment[i] / (1 - pow(_alfa, epoch + 1))).array().sqrt() + _epsilon).array()));
//         }
//     }
// };