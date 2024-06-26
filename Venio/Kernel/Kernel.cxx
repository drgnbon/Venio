#include "Kernel.hxx"

Matrixd Kernel::multiplyMatrices(const Matrixd &a, const Matrixd &b)
{
    return a * b;
}
// Matrixd Kernel::divideMatrices(const Matrixd &a, const Matrixd &b); // not supported

Matrixd Kernel::addMatrices(const Matrixd &a, const Matrixd &b)
{
    return a + b;
}
Matrixd Kernel::subtractMatrices(const Matrixd &a, const Matrixd &b)
{
    return a - b;
}
Matrixd Kernel::transposeMatrix(const Matrixd &a)
{
    return a.transpose();
}
Matrixd Kernel::elementwiseMultiplyMatrices(const Matrixd &a, const Matrixd &b)
{
    return a.array() * b.array();
}
// Matrixd Kernel::elementwiseProductMatrices(); // not supported

Matrixd Kernel::applyFunctionToElements(Matrixd a, double (*foo)(double))
{
    for (size_t i = 0; i < a.cols(); ++i)
    {
        for (size_t j = 0; j < a.rows(); ++j)
        {
            a(i, j) = foo(a(i, j));
        }
    }
    return a;
}
