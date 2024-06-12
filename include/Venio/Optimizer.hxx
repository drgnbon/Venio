#ifndef Venio_OPTIMIZER_HXX
#define Venio_OPTIMIZER_HXX

#include "Eigen/Core"
#include <map>
#include <Venio/NeuralNetwork.hxx>


typedef Eigen::MatrixXd Matrixd;

class Optimizer
{
public:
    double _gamma;
    double _alfa;
    double _epsilon;

    explicit Optimizer(NeuralNetwork& network):_network{network}{};

    virtual void updateWeights(Matrixd answer,std::shared_ptr<LossFunction> _loss_function,double learning_speed ,double epoch ) = 0;

    void backPropogation( Matrixd right_answer ,std::shared_ptr<LossFunction> _loss_function);

protected:
    NeuralNetwork& _network;


};




#endif