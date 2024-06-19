class LossFunction
{
public:
    virtual double getMediumLoss(const Matrixd &activeValue, const Matrixd &rightAnswer) = 0;
    virtual Matrixd getDerivationLoss(Matrixd activeValue,Matrixd rightAnswer) = 0;
};
class SquareErrorFunction : public LossFunction
{
public:
    double getMediumLoss(const Matrixd &activeValue, const Matrixd &rightAnswer) override
    {
        double squareError = (activeValue - rightAnswer).squaredNorm();
        return squareError / static_cast<double>(activeValue.size());
    }

    Matrixd getDerivationLoss(Matrixd activeValue,Matrixd rightAnswer) override
    {
        return 2.0 * (activeValue - rightAnswer);
    }
};