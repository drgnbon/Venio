#include "Optimizer.hxx"

class Adagrad : public Optimizer
{
public:
    explicit Adagrad(Model &network) : Optimizer(network)
    {
        _epsilon = 1e-8;
        _squared_gradient = new Matrixd[_network.getLayersSize()];
        for(int i  =0;i < _network.getLayersSize();++i)
        {
            _squared_gradient[i] = network.getLayerWeights(i);
            _squared_gradient[i].setZero();
        }
    }
    
    ~Adagrad();
    void updateWeights(double learning_speed, double epoch) override;
private:
    Matrixd* _squared_gradient;
    double _epsilon;
};
