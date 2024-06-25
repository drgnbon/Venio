#include "Optimizer.hxx"

class Adadelta : public Optimizer
{
public:
    explicit Adadelta(Model &network) : Optimizer(network)
    {
        _gamma = 0.9;
        _epsilon = 1e-8;
        _squared_gradient = new Matrixd[network.getLayersSize()];
        _squared_updates = new Matrixd[network.getLayersSize()];
        for(int i  =0;i < network.getLayersSize();++i)
        {
            _squared_updates[i] = _network.getLayerWeights(i);;
            _squared_gradient[i] = _network.getLayerWeights(i);;
            _squared_updates[i].setZero();
            _squared_gradient[i].setZero();
        }
    }
    ~Adadelta();


    void updateWeights(double learning_speed, double epoch) override;
private:
    Matrixd* _squared_gradient;
    Matrixd* _squared_updates;
    double _epsilon,_gamma;
};


