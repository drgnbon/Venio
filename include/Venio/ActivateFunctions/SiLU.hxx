#ifndef Venio_SILU_HXX
#define Venio_SILU_HXX

#include <Venio/ActivateFunction.hxx>

class SiLU : public  ActivationFunction{
public:
    SiLU();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
