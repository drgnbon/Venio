#include "Optimizer.hxx"

class RMSProp : public Optimizer
{
public:
    explicit RMSProp(Model &network) : Optimizer(network)
    {
        _epsilon = 1e-8;
        _gamma = 0.9;
        _squared_gradient = new Matrixd[network.getLayersSize()];
        for(int i = 0;i < network.getLayersSize();++i)
        {
            _squared_gradient[i] = network.getLayerWeights(i);
            _squared_gradient[i].setZero();
        }
    }
    ~RMSProp();
    void updateWeights(double learning_speed, double epoch) override;
private:
    Matrixd* _squared_gradient;
    double _gamma,_epsilon;


};

