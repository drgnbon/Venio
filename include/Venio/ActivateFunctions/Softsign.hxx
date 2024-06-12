#ifndef Venio_SOFTSIGN_HXX
#define Venio_SOFTSIGN_HXX

#include <Venio/ActivateFunction.hxx>

class Softsign : public  ActivationFunction{
public:
    Softsign();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
