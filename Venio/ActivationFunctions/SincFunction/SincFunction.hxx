#include "ActivationFunction.hxx"

class SincFunction : public ActivationFunction
{
public:
    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};