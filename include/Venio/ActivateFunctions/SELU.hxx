#ifndef Venio_SELU_HXX
#define Venio_SELU_HXX

#include <Venio/ActivateFunction.hxx>

class SELU : public  ActivationFunction{
public:
    SELU();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
