#include "ActivationFunction.hxx"

class ISRLU : public ActivationFunction
{
public:
    ISRLU() = default;
    virtual ~ISRLU() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};