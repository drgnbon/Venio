#ifndef Venio_ISRLU_HXX
#define Venio_ISRLU_HXX

#include <Venio/ActivateFunction.hxx>

class ISRLU : public  ActivationFunction{
public:
    ISRLU();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
