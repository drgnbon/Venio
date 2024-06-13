
#include <Venio/Venio.hxx>

int main() {
    Eigen::setNbThreads(6);
  NeuralNetwork network;
  network.setLayers<PerceptronLayer>({10000,5,2});
  network.setActivationFunction<Sinc>();

  TrainerStrategy trainer(network);
  trainer.setLossFunction<SquareError>();
  trainer.setOptimizer<ADAM>(network);
  //trainer.setHyperparameters(0.999,0.9,1e-8);

  Logging log;
  log.startLogging(trainer);

  Branch branch(10000,2);
  branch.generateRandomBranch(10,0,0.9);
  branch.buildBranch();

  trainer.fit(branch, 10000,0.002,false);

  log.stopLogging();
}


