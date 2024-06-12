#ifndef Venio_BENTI_HXX
#define Venio_BENTI_HXX

#include <Venio/ActivateFunction.hxx>

class Benti : public  ActivationFunction {
public:
    Benti();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
