#ifndef Venio_SIN_HXX
#define Venio_SIN_HXX

#include <Venio/ActivateFunction.hxx>

class Sin : public  ActivationFunction{
public:
    Sin();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
