#ifndef Venio_ISRU_HXX
#define Venio_ISRU_HXX

#include <Venio/ActivateFunction.hxx>

class ISRU : public  ActivationFunction{
public:
    ISRU();

    double getActivateValue(double value) override;
    double getDerivateValue(double value) override;
};


#endif
