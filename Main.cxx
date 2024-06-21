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
#include "GPUCalculation.cu"

typedef Eigen::MatrixXd Matrixd;

int main()
{

    std::cout << "hallo" < "\n";
    system("pause");
    // Eigen::setNbThreads(12);
    // LogisticFunction logistic; // pass
    // LinearFunction linear;     // pass
    // SoftSignFunction ssf;      // pass
    // SincFunction sinc;         // maybe problem
    // GhFunction gh;             // pass

    // SquareErrorFunction square; // in progress

    // // Matrixd in(1, 3);
    // // in << 0.1,0.5,0.2;
    // // Matrixd out(1, 3);
    // // out << 0.2,0.3,0.4;

    // // std::cout << square.getMediumLoss(in, out);
    // // std::cout << square.getDerivationLoss(in,out);

    // // system("pause");

    // // srand(time(NULL));

    // std::vector<std::shared_ptr<Layer>> layers{
    //     std::make_shared<SequentialLayer>(100, &ssf),
    //     std::make_shared<SequentialLayer>(500000, &ssf),
    //     std::make_shared<SequentialLayer>(1, &linear),

    // };

    // Model network(&square, layers);

    // Matrixd a(1, 100);
    // a.setConstant(0.1);
    // Matrixd b(1, 1);
    // b.setConstant(0.1);

    // network.setInput(a);

    // GD gd(network);
    // ADAM adam(network);

    // int epoch = 1;

    // while (true)
    // {

    //     network.forwardPropogation();
    //     network.backPropogation(b);

    //     // for (int i = 1; i < network.getLayersSize(); ++i)
    //     // {
    //     //     std::cout << network.getLayerWeights(i) << "\t";
    //     //     std::cout << network.getLayerWeightsGradient(i) << "\n";
    //     // }
    //     // std::cout << "\n\n";

    //     gd.updateWeights(0.00001, epoch);

    //     std::cout << network.getOutput() << "\n";
    //     //  system("pause");
    //     ++epoch;
}
}
