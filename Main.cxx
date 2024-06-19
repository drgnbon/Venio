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
        std::make_shared<SequentialLayer>(1, &linear),
        std::make_shared<SequentialLayer>(1, &linear),
        std::make_shared<SequentialLayer>(1, &linear),

    };

    Model network(&square, layers);

    Matrixd a(1, 1);
    a.setConstant(0.125);
    Matrixd b(1, 1);
    b.setConstant(0.1);

    GD gd(network);
    ADAM adam(network);

    int epoch = 1;
    network.setInput(a);

    while (true)
    {

        network.forwardPropogation();


        std::cout <<network.getLayerActiveValues(0) << "\n\n";
        std::cout <<network.getLayerActiveValues(1) <<"\t" << network.getLayerWeights(1) << "\t" <<network.getLayerBias(1) <<  "\n\n";
        std::cout <<network.getLayerActiveValues(2) <<"\t" << network.getLayerWeights(2) << "\t" <<network.getLayerBias(2) <<  "\n\n";

        network.backPropogation(b);
        gd.updateWeights(0.2, epoch);

        std::cout << network.getOutput() << "\n";
        system("pause");
        ++epoch;
    }
}
