#include "../ActivationFunction/ActivationFunction.hxx"

class LogisticFunction : public ActivationFunction{
public:
    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};