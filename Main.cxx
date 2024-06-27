#define CPU_OPTIMIZATION
#include "Venio\Venio.hxx"
#include <iostream>


// To Do
/*
1. Add Kernel to optimizers
{
    adadelta
    adagrad
    adam
    bfgs (!!!!!Do only Andrey!!!!!)
    rmsprop
}
2.Add Bias to optimizers{
    adadelta
    adagrad
    adam
    bfgs (!!!!!Do only Andrey!!!!!)
    gd
    rmsprop
}
3.Add Kernel to Model -
4.Add Kernel to Loss Function -
5.Add Kernel to ActivationFunctions -


*/
// To Do



int main()
{

    /*Matrixd a(1, 1);
    a.setConstant(0.1);
    Matrixd b(1, 1);
    b.setConstant(0.1);

    Kernel::sum(a, b);*/

    BenchMark::benchSequentialLayer();


    // LogisticFunction logistic; // pass
    // SquareErrorFunction square;

    // std::vector<std::shared_ptr<Layer>> layers{
    //     std::make_shared<SequentialLayer>(400, &logistic),
    //     std::make_shared<SequentialLayer>(200, &logistic),
    //     std::make_shared<SequentialLayer>(1, &logistic),
    // };
    // Model network(&square, layers);
    // Matrixd a(1, 400);
    // a.setConstant(0.1);
    // Matrixd b(1, 1);
    // b.setConstant(0.1);

    // GD f(network);

    // int epoch = 1;

    // while (true)
    // {

    //     network.forwardPropogation();
    //     network.backPropogation(b);
    //     f.updateWeights(0.005, epoch);
    //     std::cout << network.getOutput() << "\n";
    //     ++epoch;
    // }

    //     ArcTg at;
    //     Benti benti;
    //     ELU elu;
    //     GH gh;
    //     ISRLU isrlu;
    //     ISRU isru;
    //     LinearFunction linear;
    //     LogisticFunction logistic; // pass
    //     LReLU lrelu;
    //     ReLU relu;
    //     SELU selu;
    //     SiLU silu;
    //     Sin sin;
    //     SincFunction sinc;
    //     SoftPlus sp;
    //     SoftSignFunction ss;
    //     TH th;

    //     SquareErrorFunction square;

    //     std::vector<std::shared_ptr<Layer>> layers{
    //         std::make_shared<SequentialLayer>(20, &linear),
    //         std::make_shared<SequentialLayer>(10, &at),
    //         std::make_shared<SequentialLayer>(10, &benti),
    //         std::make_shared<SequentialLayer>(10, &elu),
    //         std::make_shared<SequentialLayer>(10, &gh),
    //         std::make_shared<SequentialLayer>(10, &isrlu),
    //         std::make_shared<SequentialLayer>(10, &isru),
    //         std::make_shared<SequentialLayer>(10, &linear),
    //         std::make_shared<SequentialLayer>(10, &logistic),
    //         std::make_shared<SequentialLayer>(10, &lrelu),
    //         std::make_shared<SequentialLayer>(10, &relu),
    //         std::make_shared<SequentialLayer>(10, &selu),
    //         std::make_shared<SequentialLayer>(10, &silu),
    //         std::make_shared<SequentialLayer>(10, &logistic),
    //         std::make_shared<SequentialLayer>(10, &sin),
    //         std::make_shared<SequentialLayer>(10, &sinc),
    //         std::make_shared<SequentialLayer>(10, &sp),
    //         std::make_shared<SequentialLayer>(10, &ss),
    //         std::make_shared<SequentialLayer>(10, &th),
    //         std::make_shared<SequentialLayer>(1, &linear),
    //     };

    //     Model network(&square, layers);

    //     Matrixd a(1, 20);
    //     a.setConstant(0.1);
    //     Matrixd b(1, 1);
    //     b.setConstant(0.1);

    //     network.setInput(a);

    //     // Check indexation in optimizer very important

    //     GD f(network);
    //     // ADAM f(network);
    //     // BFGS f(network);
    //     // Adadelta f(network);
    //     // Adagrad f(network);
    //     // RMSProp f(network);

    //     int epoch = 1;

    //     while (true)
    //     {

    //         network.forwardPropogation();
    //         network.backPropogation(b);
    //         f.updateWeights(0.005, epoch);
    //         std::cout << network.getOutput() << "\n";
    //         ++epoch;
    //     }
}