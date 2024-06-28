#pragma once

#include <vector>
#include <thread>
#include <Eigen\Core>
typedef Eigen::MatrixXd Matrixd;
typedef Eigen::ArrayXXd Arrayd;

namespace Kernel
{



    Matrixd multiply(const Matrixd &a, const Matrixd &b);

    Arrayd divideArrays(const Arrayd& a, const Arrayd& b);

    Matrixd scalarDivide(const Matrixd& a, double s);

    Arrayd scalarSum(Arrayd a, double s);


    Matrixd sum(const Matrixd& a, const Matrixd& b);
    Matrixd sub(const Matrixd &a, const Matrixd &b);
    Matrixd transpose(const Matrixd &a);
    Matrixd eMultiply(const Matrixd &a, const Matrixd &b);
    //Matrixd elementwiseProductMatrices(); // not supported
    Matrixd scalarMultiply(Matrixd a, double s);
    Matrixd scalarMultiply(double s, Matrixd a);
    double dot(Eigen::VectorXd a, Eigen::VectorXd b);


    Arrayd scalarSum(Arrayd a, double s);

    Matrixd applyFunctionToElements(Matrixd a, double (*foo)(double));


};