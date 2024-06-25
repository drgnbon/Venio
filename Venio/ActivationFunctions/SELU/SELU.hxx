#include "ActivationFunction.hxx"

class SELU : public ActivationFunction
{
public:
    SELU() = default;
    virtual ~SELU() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};