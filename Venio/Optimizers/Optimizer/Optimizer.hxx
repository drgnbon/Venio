

class Optimizer
{
protected:
    Model &_network;

public:
    explicit Optimizer(Model &network) : _network{network}
    {
        _network = network;
    }
    virtual void updateWeights(double learning_speed, double epoch) = 0;
};