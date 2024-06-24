#include "Venio\Venio.hxx"
#include <iostream>

int main()
{
    LogisticFunction logistic; // pass
    SquareErrorFunction square;


    std::vector<std::shared_ptr<Layer>> layers{
            std::make_shared<SequentialLayer>(100, &logistic),
            std::make_shared<SequentialLayer>(500, &logistic),
            std::make_shared<SequentialLayer>(1, &logistic),
    };

    Model network(&square, layers);

    Matrixd a(1, 100);
    a.setConstant(0.1);
    Matrixd b(1, 1);
    b.setConstant(0.1);

    network.setInput(a);
    while (true)
    {
        network.forwardPropogation();
        network.backPropogation(b);
        std::cout << network.getOutput() << "\n";
    }
}