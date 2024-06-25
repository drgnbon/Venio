#include "ActivationFunction.hxx"

class Sin : public ActivationFunction
{
public:
    Sin() = default;
    virtual ~Sin() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};