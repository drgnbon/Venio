// #include "GD.hxx"

//class GD : public Optimizer
//{
//public:
//    explicit GD(Model& network) : Optimizer(network) {}

 //   void updateWeights(double learning_speed = 0.05, double epoch = 1) override
 //   {
 //       for (int i = 1; i < _network.getLayersSize(); i++)
 //       {
  //          _network.setLayerWeights(i, _network.getLayerWeights(i) - _network.getLayerWeightsGradient(i) * learning_speed);
  //          _network.setLayerBias(i, _network.getLayerBias(i) - _network.getLayerBiasGradient(i) * learning_speed);
 //       }
 //   }
//};