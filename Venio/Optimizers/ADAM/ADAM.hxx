#pragma once
#include "Optimizer.hxx"

 class ADAM : public Optimizer
 {
 public:
     explicit ADAM(Model& network) : Optimizer(network)
     {   
         _network = network;
         size_t layer_count = _network.getLayersSize();

         _history_speed_weights = new Matrixd[layer_count];
         _history_moment_weights = new Matrixd[layer_count];
         _history_speed_bias = new Matrixd[layer_count];
         _history_moment_bias = new Matrixd[layer_count];

         for (size_t i = 1; i < layer_count; ++i)
         {
             _history_speed_weights[i] = _network.getLayerWeights(i).setZero();
             _history_moment_weights[i] = _history_speed_weights[i];
             _history_speed_bias[i] = _network.getLayerBias(i).setZero();
             _history_moment_bias[i] = _history_speed_bias[i];
         }
     }

     ~ADAM()
     {
         delete[] _history_speed_weights;
         delete[] _history_moment_weights;
         delete[] _history_speed_bias;
         delete[] _history_moment_bias;
     }

     void updateWeights(double learning_speed, double epoch) override;

 private:
     Matrixd* _history_speed_weights;
     Matrixd* _history_moment_weights;
     Matrixd* _history_speed_bias;
     Matrixd* _history_moment_bias;

     double _gamma = 0.9;
     double _alfa = 0.999;
     double _epsilon = 1e-8;
 };




