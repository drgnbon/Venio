#include "ActivationFunction.hxx"

class SoftSignFunction : public ActivationFunction{
public:
    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};