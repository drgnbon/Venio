#ifndef Venio_SOFTPLUS_HXX
#define Venio_SOFTPLUS_HXX

#include <Venio/ActivateFunction.hxx>

class SoftPlus : public  ActivationFunction{
public:
    SoftPlus();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
