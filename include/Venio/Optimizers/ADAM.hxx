#ifndef FENDL_ADAM_HXX
#define FENDL_ADAM_HXX

#include <Venio//Optimizer.hxx>

class ADAM : public Optimizer{
public:
    explicit ADAM(NeuralNetwork& network);
    ~ADAM();
    void updateWeights(Matrixd answer,std::shared_ptr<LossFunction> _loss_function,
                       double learning_speed,double epoch) override;

private:
    Matrixd*  _history_speed;
    Matrixd* _history_moment;
};


#endif


