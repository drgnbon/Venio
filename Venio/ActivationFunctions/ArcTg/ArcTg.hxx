#include "ActivationFunction.hxx"

class ArcTg : public ActivationFunction
{
public:
    ArcTg() = default;
    virtual ~ArcTg() = default;

    double toActivateValue(double x) override;
    double toDerivateValue(double x) override;
};