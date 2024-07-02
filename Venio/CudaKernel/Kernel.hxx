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
typedef Eigen::VectorXd Vectord;


namespace Kernel
{

    //CUDA DEBUG{
    inline void checkCuda(cudaError_t result, const char* msg);
    inline void checkCublas(cublasStatus_t result, const char* msg);
    inline cublasHandle_t getCublasHandle(void* handle) noexcept;
    // ---------}






    //Only matrix{
    Matrixd matrixMultiply(const Matrixd& A, const Matrixd& B);
    Matrixd eMultiply(const Matrixd& a, const Matrixd& b);
    Matrixd scalarMultiply(Matrixd a, double s);
    Matrixd scalarMultiply(double s,Matrixd a);//remoove it
    Matrixd scalarDivide(const Matrixd& a, double s);
    Matrixd sum(const Matrixd& a, const Matrixd& b);
    Matrixd sub(const Matrixd& a, const Matrixd& b);
    Matrixd transpose(const Matrixd& a);
    
    
    Matrixd applyFunctionToElements(Matrixd a, double (*foo)(double));
    
    // -------- }


    //Only vector{
    double dot(Vectord a, Vectord b);
    Vectord vectorTranspose(const Vectord& a);
    Matrixd vectorMultiply(const Vectord& a, const Vectord& b);

    // -------- }


    //Only arrays{

    Arrayd sqr(const Arrayd& a);
    Arrayd sqrt(const Arrayd& a);
    Arrayd scalarAdd(const Arrayd& a, double b);
    Arrayd multiplyArrays(const Arrayd& a, const Arrayd& b);
    Arrayd scalarArrayMultiply(const Arrayd& a, double s);
    Arrayd scalarArrayMultiply(double s, const Arrayd& a);
    Arrayd divideArrays(const Arrayd& a, const Arrayd& b);

    Arrayd scalarSum(Arrayd a, double s);

    Arrayd scalarSum(Arrayd a, double s);

    // -------- }


    //others{




    // -------- }
    


};