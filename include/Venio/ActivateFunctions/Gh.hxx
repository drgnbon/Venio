#ifndef Venio_GH_HXX
#define Venio_GH_HXX

#include <Venio/ActivateFunction.hxx>

class Gh : public  ActivationFunction{
public:
    Gh();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
