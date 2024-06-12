#ifndef Venio_LINEAR_HXX
#define Venio_LINEAR_HXX

#include <Venio/ActivateFunction.hxx>

class Linear : public  ActivationFunction{
public:
    Linear();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
