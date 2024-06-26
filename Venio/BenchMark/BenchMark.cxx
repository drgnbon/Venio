#include <BenchMark.hxx>

void BenchMark::benchSequentialLayer()
{
    double learning_speed = 0.001;
    double less_aim = 0.01;

    LogisticFunction logistic;
    SquareErrorFunction square;
    std::vector<std::shared_ptr<Layer>> layers{
        std::make_shared<SequentialLayer>(5000, &logistic),
        std::make_shared<SequentialLayer>(2500, &logistic),
        std::make_shared<SequentialLayer>(1250, &logistic),
        std::make_shared<SequentialLayer>(625, &logistic),
        std::make_shared<SequentialLayer>(1, &logistic),
    };

    std::string structure = "[";
    for (auto i : layers) {
        structure += std::to_string(i->getLayerSize()) + ",";
    }
    structure.pop_back();
    structure += "]";

    Model network(&square, layers);
    Matrixd input = Matrixd::Constant(1, 5000, 0.1);
    Matrixd right_answer = Matrixd::Constant(1, 1, 0.5);
    network.setInput(input);
    GD optimizer(network);

    HANDLE hProcess = GetCurrentProcess();
    PROCESS_MEMORY_COUNTERS pmc;
    double sum_tps = 0;
    int iteration = 1;

    while (true)
    {
        clock_t old_time = clock();
        double old_loss = network.getAverageLoss(right_answer);

        // Perform calculations
        network.forwardPropogation();
        network.backPropogation(right_answer);
        optimizer.updateWeights(learning_speed, iteration);
        // End of calculations

        // Logging
        clock_t new_time = clock();
        sum_tps += static_cast<double>(new_time - old_time) / CLOCKS_PER_SEC;
        double new_loss = network.getAverageLoss(right_answer);

        system("cls");
        printf("Iteration: %d \n",iteration);
        printf("Time for test: %ld (ms) | TPS: %.3g (%.3g avg)\n",new_time-old_time,1.0/(double(new_time-old_time)/1000.0),1.0/(sum_tps/iteration));
        double loss_change = old_loss - new_loss;
        double loss_change_percentage = (loss_change / old_loss) * 100.0;
        printf("loss on output layer: %.5f | Loss change: %.5f (%.3f%% less)\n",new_loss,loss_change,loss_change_percentage);
        printf("Learning speed: %.5f | Structure: ",learning_speed);
        std::cout << structure <<"\n\n";
        // Get memory usage
        GetProcessMemoryInfo(hProcess, &pmc, sizeof(pmc));
        printf("Memory usage: %llu (MB) | Max: %llu (MB)",pmc.WorkingSetSize / (1024 * 1024),pmc.PeakWorkingSetSize / (1024 * 1024));
        // Get memory usage
        //Logging
        ++iteration;

        if(new_loss < less_aim)
        {
            system("cls");
            printf("Testing results:\n");
            printf("\tLearning speed: %.5f | Structure: ",learning_speed);
            std::cout << structure <<"\n";
            printf("\tMax memory usage: %llu (MB)\n",pmc.PeakWorkingSetSize / (1024 * 1024));
            printf("\tIterations for education: %d | Time for education %.1f(sec.) %.1f(min.)\n",iteration,sum_tps,sum_tps/60);
            printf("\tAverage TPS: %f\n",1.0/(sum_tps/iteration));


            system("pause");
            CloseHandle(hProcess);
            return;
        }
    }

}