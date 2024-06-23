#include "../ActivationFunction/ActivationFunction.hxx"

class GhFunction : public ActivationFunction{
public:
    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};