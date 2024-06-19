#include <Eigen\Core>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>

#include "RandomGenerator.cxx"
#include "ErrorLogger.cxx"
#include "ActivationFunction.cxx"
#include "LossFunction.cxx"
#include "Layer.cxx"
#include "Model.cxx"
#include "Optimizer.cxx"

using namespace Eigen;
typedef Eigen::MatrixXd Matrixd;

int main()
{

    // srand(time(NULL));

    SquareErrorFunction square;
    LogisticFunction logistic;
    LinearFunction linear;
    SoftSignFunction ssf;
    SincFunction sinc;
    GhFunction gh;

    std::vector<std::shared_ptr<Layer>> layers{
        std::make_shared<SequentialLayer>(6, &ssf),
        std::make_shared<SequentialLayer>(3000, &ssf),
        std::make_shared<SequentialLayer>(2, &ssf),

    };

    Model network(&square, layers);

    Matrixd a(1, 6);
    a << 0.7, 0.7, 0.7, 0.7, 0.7, 0.7;
    Matrixd b(1, 2);
    b << 0.3, 0.3;

    GD gd(network);
    ADAM adam(network);

    //int epoch = 1;

    while (true)
    {
        network.setInput(a);
        network.forwardPropogation();
        //network.backPropogation(b);
        //gd.updateWeights(0.2, epoch);

        std::cout << network.getOutput() << "\n";
        //// system("pause");
        //++epoch;
    }
}
