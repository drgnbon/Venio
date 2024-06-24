#include "ActivationFunction.hxx"

class LogisticFunction : public ActivationFunction
{
public:
    LogisticFunction() = default;
    virtual ~LogisticFunction() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};