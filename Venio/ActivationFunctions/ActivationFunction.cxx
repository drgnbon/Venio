#include "ActivationFunction.hxx"

Matrixd ActivationFunction::toActivateMatrix(Matrixd matrix)
{
    for (int i = 0; i < matrix.rows(); ++i)
    {
        for (int j = 0; j < matrix.cols(); ++j)
        {
            matrix(i, j) = toActivateValue(matrix(i, j));
        }
    }
    return matrix;
}
Matrixd ActivationFunction::toDerivateMatrix(Matrixd matrix)
{
    for (int i = 0; i < matrix.rows(); ++i)
    {
        for (int j = 0; j < matrix.cols(); ++j)
        {
            matrix(i, j) = toDerivateValue(matrix(i, j));
        }
    }
    return matrix;
}