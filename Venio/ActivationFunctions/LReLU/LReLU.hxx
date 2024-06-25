#include "ActivationFunction.hxx"

class LReLU : public ActivationFunction
{
public:
    LReLU() = default;
    virtual ~LReLU() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};