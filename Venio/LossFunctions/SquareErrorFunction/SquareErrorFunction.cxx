#include "SquareErrorFunction.hxx"

double SquareErrorFunction::getMediumLoss(const Matrixd &activeValue, const Matrixd &rightAnswer)
{
    double squareError = (activeValue - rightAnswer).squaredNorm();
    return squareError / static_cast<double>(activeValue.size());
}
Matrixd SquareErrorFunction::getDerivationLoss(Matrixd activeValue, Matrixd rightAnswer)
{
    return (2.0 / double(activeValue.cols())) * (activeValue - rightAnswer);
}
