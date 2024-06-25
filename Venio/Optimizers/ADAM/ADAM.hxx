 #include "Optimizer.hxx"

 class ADAM : public Optimizer
 {
 public:
     Matrixd *_history_speed;
     Matrixd *_history_moment;
     double _gamma = 0.9;
     double _alfa = 0.999;
     double _epsilon = 1e-8;

     explicit ADAM(Model& network) : Optimizer(network)
     {
         _network = network;
         _history_speed = new Matrixd[network.getLayersSize()];
         _history_moment = new Matrixd[network.getLayersSize()];

         for (int i = 1; i < network.getLayersSize(); ++i)
         {
             _history_speed[i] = network.getLayerWeights(i);
             _history_moment[i] = network.getLayerWeights(i);
             _history_speed[i].setZero();
             _history_moment[i].setZero();
         }
     }

     ~ADAM()
     {
         delete[] _history_speed;
         delete[] _history_moment;
     }

     void updateWeights(double learning_speed, double epoch) override;
 };




