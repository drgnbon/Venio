//
// Created by Andrey on 12.03.2024.
//

#ifndef Venio_RELU_HXX
#define Venio_RELU_HXX
#include <Venio/ActivateFunction.hxx>

class ReLU : public ActivationFunction {
public:
	ReLU();

	double getActivateValue(double value) override;
	double getDerivateValue(double value) override;


};



#endif //Venio_RELU_HXX
