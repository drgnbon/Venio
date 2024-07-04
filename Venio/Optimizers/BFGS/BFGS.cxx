#include "BFGS.hxx"

void BFGS::updateWeights(double learning_speed, double epoch)
{
    for(int i = 1;i < _network.getLayersSize();i++){

        _layer_weights_gradient = _network.getLayerWeightsGradient(i);
        _layer_weights = _network.getLayerWeights(i);

        _old_gradient[i] = Eigen::Map<Vector>(_layer_weights_gradient.data(),
                                              _layer_weights_gradient.size());

        _old_weights[i] = Eigen::Map<Vector>(_layer_weights.data(),
                                             _layer_weights.size());

#ifdef GPU_OPTIMIZATION
        _new_weights[i] = K::subVV(_old_weights[i], K::multVS(learning_speed,K::multMV(_inversed_hessian[i], _old_gradient[i])));
        
#endif    
#ifdef CPU_OPTIMIZATION
        _new_weights[i] = _old_weights[i] - learning_speed*(_inversed_hessian[i] * _old_gradient[i]);
#endif


        _network.setLayerWeights(i,Eigen::Map<Matrixd>(_new_weights[i].data(),
                                                       _layer_weights.rows(),
                                                       _layer_weights.cols()));
    }
    _network.forwardPropogation();

    //WARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    _network.backPropogation();

    for(int i = 1;i < _network.getLayersSize();i++){

        _new_gradient[i] = Eigen::Map<Vector>(_network.getLayerWeightsGradient(i).data(),
                                              _network.getLayerWeightsGradient(i).size());


#ifdef GPU_OPTIMIZATION
        s = K::subVV(_new_weights[i],_old_weights[i]);
        y = K::subVV(_new_gradient[i],_old_gradient[i]);
        ro = 1 / K::dotVV(y, s);

        Matrixd a = K::subMM(I[i], K::multMS(ro, K::multMM(s.matrix(), K::transposeV(y))));//shit from Eigen
        Matrixd b = K::subMM(I[i], K::multMS(ro, K::multMM(y.matrix(), K::transposeV(s))));//shit from Eigen

        _inversed_hessian[i] = K::sumMM(K::multMM(K::multMM(a, _inversed_hessian[i]), b) , K::multMS(ro, K::multMM(s.matrix(), K::transposeV(s))));
#endif


#ifdef CPU_OPTIMIZATION
        s = _new_weights[i] - _old_weights[i];
        y = _new_gradient[i] - _old_gradient[i];
        ro = 1 / y.dot(s);

        Matrixd a = I[i] - ro * (s * y.transpose());
        Matrixd b = I[i] - ro * (y * s.transpose());


        _inversed_hessian[i] =  a * _inversed_hessian[i] * b  + ro * s * s.transpose();
#endif

        


        
    }
}

BFGS::~BFGS() {
    delete[] _inversed_hessian;
    delete[] I;
    delete[] _old_gradient;
    delete[] _new_gradient;
    delete[] _old_weights;
    delete[] _new_weights;

}
