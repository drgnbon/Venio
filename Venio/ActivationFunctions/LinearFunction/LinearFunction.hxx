#include "../ActivationFunction/ActivationFunction.hxx"

class LinearFunction : public ActivationFunction{
public:
    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};