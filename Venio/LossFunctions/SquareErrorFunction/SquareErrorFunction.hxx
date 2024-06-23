
#include "../LossFunction/LossFunction.hxx"

class SquareErrorFunction : public LossFunction
{
public:
    double getMediumLoss(const Matrixd &activeValue, const Matrixd &rightAnswer) override;
    Matrixd getDerivationLoss(Matrixd activeValue, Matrixd rightAnswer) override;
};