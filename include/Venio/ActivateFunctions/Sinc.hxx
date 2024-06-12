#ifndef Venio_SINC_HXX
#define Venio_SINC_HXX

#include <Venio/ActivateFunction.hxx>

class Sinc : public  ActivationFunction{
public:
    Sinc();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif