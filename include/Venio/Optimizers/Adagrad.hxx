#ifndef FENDL_ADAGRAD_HXX
#define FENDL_ADAGRAD_HXX

#include <Venio/Optimizer.hxx>


class Adagrad : public Optimizer{
public:
    explicit Adagrad(NeuralNetwork& network);
    ~Adagrad();
    void updateWeights(Matrixd answer,std::shared_ptr<LossFunction> _loss_function,
                       double learning_speed,double epoch) override;
private:
    Matrixd* _squared_gradient;
};


#endif
