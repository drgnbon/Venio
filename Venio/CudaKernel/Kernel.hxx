#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

#include <vector>
#include <thread>
#include <Eigen\Core>
typedef Eigen::MatrixXd Matrixd;
typedef Eigen::ArrayXXd Arrayd;



namespace Kernel
{
    inline void checkCuda(cudaError_t result, const char* msg);

    inline void checkCublas(cublasStatus_t result, const char* msg);

    inline cublasHandle_t getCublasHandle(void* handle) noexcept;

    Matrixd multiply(const Matrixd& A, const Matrixd& B);

    Arrayd divideArrays(const Arrayd& a, const Arrayd& b);

    Matrixd scalarDivide(const Matrixd& a, double s);

    Arrayd scalarSum(Arrayd a, double s);


    Matrixd sum(const Matrixd& a, const Matrixd& b);
    Matrixd sub(const Matrixd& a, const Matrixd& b);
    Matrixd transpose(const Matrixd& a);
    Matrixd eMultiply(const Matrixd& a, const Matrixd& b);
    //Matrixd elementwiseProductMatrices(); // not supported
    Matrixd scalarMultiply(Matrixd a, double s);
    Matrixd scalarMultiply(double s, Matrixd a);
    double dot(Eigen::VectorXd a, Eigen::VectorXd b);

    Arrayd sqr(const Arrayd& a);
    Arrayd sqrt(const Arrayd& a);
    Arrayd scalarAdd(const Arrayd& a,double b);
    Arrayd multiplyArrays(const Arrayd& a, const Arrayd&  b);
    Matrixd scalarArrayMultiply(const Arrayd& a, double s);
    Matrixd scalarArrayMultiply(double s, const Arrayd& a);


    Arrayd scalarSum(Arrayd a, double s);

    Matrixd applyFunctionToElements(Matrixd a, double (*foo)(double));


};