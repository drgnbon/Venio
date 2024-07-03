//#pragma once
//#include "Optimizer.hxx"
//
//typedef Eigen::VectorXd Vector;
//
//class BFGS : public Optimizer
//{
//public:
//    explicit BFGS(Model& network) : Optimizer(network){
//        _inversed_hessian = new Matrixd[network.getLayersSize()];
//        I = new Matrixd[network.getLayersSize()];
//        _old_gradient = new Vector[network.getLayersSize()];
//        _new_gradient = new Vector[network.getLayersSize()];
//        _old_weights  = new Vector[network.getLayersSize()];
//        _new_weights  = new Vector[network.getLayersSize()];
//        ro = 0;
//
//        for(int i  = 0;i < network.getLayersSize();++i)
//        {
//            _inversed_hessian[i] = Matrixd::Identity(network.getLayerWeights(i).size(),
//                                                    network.getLayerWeights(i).size());
//
//            I[i] =  _inversed_hessian[i];
//            _old_gradient[i] = Vector::Zero(network.getLayerWeights(i).size());
//            _new_gradient[i] = Vector::Zero(network.getLayerWeights(i).size());
//            _old_weights[i]  = Vector::Zero(network.getLayerWeights(i).size());
//            _new_weights[i]  = Vector::Zero(network.getLayerWeights(i).size());
//        }
//
//    }
//    ~BFGS();
//    void updateWeights(double learning_speed, double epoch) override;
//private:
//    Matrixd _layer_weights, _layer_weights_gradient;
//    Matrixd *_inversed_hessian,*I;
//    Vector y,s,*_old_gradient,*_old_weights,*_new_weights,*_new_gradient;
//    double ro;
//};
//
//
//
