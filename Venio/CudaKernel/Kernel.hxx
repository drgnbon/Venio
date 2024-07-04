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
    Matrixd multMM(const Matrixd& A, const Matrixd& B);

    Matrixd emultMM(const Matrixd& a, const Matrixd& b);

    Matrixd multMS(double S, const Matrixd& A);
    Matrixd multMS(const Matrixd& A, double S);

    Matrixd divMS(const Matrixd& a, double s);
    Matrixd divMS(double s, const Matrixd& a);

    Matrixd sumMM(const Matrixd& a, const Matrixd& b);

    Matrixd subMM(const Matrixd& A, const Matrixd& B);

    Matrixd transposeM(const Matrixd& a); 
    
    Matrixd applyFuncMF(Matrixd a, double (*foo)(double));
    Matrixd applyFuncMF(double (*foo)(double), Matrixd a);
    // -------- }



    //Only vector{
    Vectord multVS(const Vectord& A, double S);
    Vectord multVS(double S, const Vectord& A);

    Vectord subVV(const Vectord& A, const Vectord& B);


    double dotVV(const Vectord& a, const Vectord& b);

    Matrixd transposeV(const Eigen::MatrixXd& A);//shit from Eigen,there need be Vector!!!But Eigen think else

    Matrixd multVV(const Vectord& a, const Vectord& b);
    // -------- }


    //Only arrays{
    Arrayd sqrA(const Arrayd& a);

    Arrayd sqrtA(const Arrayd& a);

    Arrayd sumAA(const Arrayd& a, const Arrayd& b);

    Arrayd sumAS(const Arrayd& a, double b);
    Arrayd sumAS(double b, const Arrayd& a);

    Arrayd multAA(const Arrayd& a, const Arrayd& b);

    Arrayd multAS(const Arrayd& a, double s);
    Arrayd multAS(double s, const Arrayd& a);

    Arrayd divAA(const Arrayd& a, const Arrayd& b);
    // -------- }

    Vectord multMV(const Matrixd& A, const Vectord& B);
    Vectord multVM(const Vectord& A, const Matrixd& B);
    //others{
    // -------- }
};