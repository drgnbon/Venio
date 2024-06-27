#pragma once

#include <vector>
#include <thread>
#include <Eigen\Core>
typedef Eigen::MatrixXd Matrixd;

namespace Kernel
{



    Matrixd multiply(const Matrixd &a, const Matrixd &b);
    //Matrixd divideMatrices(const Matrixd &a, const Matrixd &b); // not supported
    Matrixd sum(const Matrixd& a, const Matrixd& b);
    Matrixd sub(const Matrixd &a, const Matrixd &b);
    Matrixd transpose(const Matrixd &a);
    Matrixd eMultiply(const Matrixd &a, const Matrixd &b);
    //Matrixd elementwiseProductMatrices(); // not supported
    Matrixd scalarMultiply(Matrixd a, double s);

    Matrixd applyFunctionToElements(Matrixd a, double (*foo)(double));


};