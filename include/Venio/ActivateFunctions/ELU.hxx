#ifndef Venio_ELU_HXX
#define Venio_ELU_HXX

#include <Venio/ActivateFunction.hxx>

class ELU : public  ActivationFunction{
public:
    ELU();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
