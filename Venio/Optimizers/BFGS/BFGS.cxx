#include "BFGS.hxx"

void BFGS::updateWeights(double learning_speed, double epoch)
{
    for(int i = 0;i < _network.getLayersSize()-1;i++){

        _old_gradient[i] = Eigen::Map<Vector>(_network.getLayerWeightsGradient(i).data(),
                                              _network.getLayerWeightsGradient(i).size());

        _old_weights[i] = Eigen::Map<Vector>(_network.getLayerWeights(i).data(),
                                             _network.getLayerWeights(i).size());

        _new_weights[i] = _old_weights[i] - learning_speed*(_inversed_hessian[i]*_old_gradient[i]);

        _network.setLayerWeights(i,Eigen::Map<Matrixd>(_new_weights[i].data(),
                                                            _network.getLayerWeights(i).rows(),
                                                            _network.getLayerWeights(i).cols())); 
    }
    _network.forwardPropogation();

    //WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    _network.backPropogation();

    for(int i = 0;i < _network.getLayersSize()-1;i++){
        _new_gradient[i] = Eigen::Map<Vector>(_network.getLayerWeightsGradient(i).data(),
                                              _network.getLayerWeightsGradient(i).size());

        s[i] = _new_weights[i]-_old_weights[i];
        y[i] = _new_gradient[i]-_old_gradient[i];
        ro = 1 / y[i].dot(s[i]);
        _inversed_hessian[i] = (I[i] - ro*s[i]*y[i].transpose())*_inversed_hessian[i]*(I[i] - ro*y[i]*s[i].transpose()) + ro*s[i]*s[i].transpose();
    }
}

BFGS::~BFGS() {
    delete[] _inversed_hessian;
    delete[] I;
    delete[] y;
    delete[] s;
    delete[] _old_gradient;
    delete[] _new_gradient;
    delete[] _old_weights;
    delete[] _new_weights;

}
