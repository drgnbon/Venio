#include "../Optimizer/Optimizer.hxx"

class GD : public Optimizer
{
public:
    explicit GD(Model &network) : Optimizer(network) {}

    void updateWeights(double learning_speed = 0.05, double epoch = 1) override;
};