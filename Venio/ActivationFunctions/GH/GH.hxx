#include "ActivationFunction.hxx"

class GH : public ActivationFunction
{
public:
    GH() = default;
    virtual ~GH() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};