#ifndef Venio_SIGMOID_HXX
#define Venio_SIGMOID_HXX

#include <Venio/ActivateFunction.hxx>


class Sigmoid :public ActivationFunction {
public:
    Sigmoid();

    double getActivateValue(double value) override;
	double getDerivateValue(double value) override;


};


#endif
