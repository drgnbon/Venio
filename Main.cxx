#include <FenDL/FenDL.hxx>
//
//int main() {
//    Eigen::setNbThreads(12);
//  NeuralNetwork network;
//  network.setLayers<PerceptronLayer>({80000,20,2});
//  network.setActivationFunction<Sigmoid>();
//
//  TrainerStrategy trainer(network);
//  trainer.setLossFunction<SquareError>();
//  trainer.setOptimizer<ADAM>(network);
//  //trainer.setHyperparameters(0.999,0.9,1e-8);
//
//  Logging log;
//  log.startLogging(trainer);
//
//  Branch branch(80000,2);
//  branch.generateRandomBranch(10,0,0.9);
//  branch.buildBranch();
//
//  trainer.fit(branch, 10000,0.001,false);
//
//  log.stopLogging();
//}
//
//
////Adadelta очень быстрый, неточный. Под конец обучения ошибка будет сильно прыгать и до конца не дойдет, не нужен learning rate
////RMSProp очень быстрый, более точный чем Adadelta. Под конец обучения ошибка будет сильно прыгать и до конца не дойдет
////Adagrad быстрый, точный, норм темка
////ADAM средний по скорости, точный, идет без вопросов к глобальному минимуму не страшны неровности
////BGFS быстрый, очень точный, огромные требования к оперативной памяти и процессору, оптимизатор 2-го порядка
////GD медленный, точный, самый базовый оптимизатор работающий на градиенте
int main()
{
    Eigen::MatrixXd mat1(3, 3);
    mat1 << 1, 2, 3,
            4, 5, 6,
            7, 8, 9;

    Eigen::MatrixXd mat2(3, 2);
    mat2 << 7, 8,
            9, 10,
            11, 12;

    Eigen::MatrixXd result = mat1.block(0, 0, 1, 1)*mat2.block(0, 0, 1, 1);


    system("pause");
    std::cout << result << std::endl;
    system("pause");
}