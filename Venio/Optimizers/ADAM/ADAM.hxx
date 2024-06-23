#include "../Optimizer/Optimizer.hxx"

class ADAM : public Optimizer
{
public:
    Matrixd *_history_speed;
    Matrixd *_history_moment;
    double _gamma = 0.9;
    double _alfa = 0.999;
    double _epsilon = 1e-8;

    ADAM(Model &network) : Optimizer(network);
    ~ADAM();

    void updateWeights(double learning_speed = 0.5, double epoch = 1);
};