#ifndef Venio_LRELU_HXX
#define Venio_LRELU_HXX

#include <Venio/ActivateFunction.hxx>

class LReLU : public  ActivationFunction{
public:
    LReLU();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
