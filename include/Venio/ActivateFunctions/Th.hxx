#ifndef Venio_TH_HXX
#define Venio_TH_HXX

#include <Venio/ActivateFunction.hxx>

class Th : public  ActivationFunction {
public:
	Th();

	double getActivateValue(double value) override;
	double getDerivateValue(double value) override;

};


#endif
