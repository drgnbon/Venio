#include "ActivationFunction.hxx"


// class ActivationFunction
// {
// public:

    

//     Matrixd toActivateMatrix(Matrixd matrix)
//     {
//         for (int i = 0; i < matrix.rows(); ++i)
//         {
//             for (int j = 0; j < matrix.cols(); ++j)
//             {
//                 matrix(i, j) = toActivateValue(matrix(i, j));
//             }
//         }
//         return matrix;
//     }
//     Matrixd toDerivateMatrix(Matrixd matrix)
//     {
//         for (int i = 0; i < matrix.rows(); ++i)
//         {
//             for (int j = 0; j < matrix.cols(); ++j)
//             {
//                 matrix(i, j) = toDerivateValue(matrix(i, j));
//             }
//         }
//         return matrix;
//     }
// };
// class LogisticFunction : public ActivationFunction
// {
// public:
//     double toActivateValue(double x) override
//     {
//         return 1.f / (1.f + exp(-x));
//     }
//     double toDerivateValue(double x) override
//     {
//         double activatedValue = toActivateValue(x);
//         return activatedValue * (1.f - activatedValue);
//     }
// };
// class LinearFunction : public ActivationFunction
// {
// public:
//     double toActivateValue(double x) override
//     {
//         return x;
//     }

//     double toDerivateValue(double x) override
//     {
//         return 1.0;
//     }
// };
// class SoftSignFunction : public ActivationFunction
// {
// public:
//     double toActivateValue(double x) override
//     {
//         return x / (1.0 + fabs(x)); // Используйте std::abs
//     }

//     double toDerivateValue(double x) override
//     {
//         return 1.0 / std::pow(1.0 + fabs(x), 2); // Используйте std::pow и std::abs
//     }
// };
// class SincFunction : public ActivationFunction
// {
// public:
//     double toActivateValue(double x) override
//     {
//         if (fabs(x) < 1e-8)
//             return 1.0;
//         return sin(x) / x;
//     }

//     double toDerivateValue(double x) override
//     {
//         if (fabs(x) < 1e-8)
//             return 0.0;
//         return (cos(x) / x) - (sin(x) / (x * x));
//     }
// };
// class GhFunction : public ActivationFunction
// {
// public:
//     double toActivateValue(double x) override
//     {
//         return exp(-(x * x));
//     }

//     double toDerivateValue(double x) override
//     {
//         return -2 * x * exp(-(x * x));
//     }
// };