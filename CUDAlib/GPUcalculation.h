#ifndef GPUCALCULATION_H
#define GPUCALCULATION_H

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <iostream>

class GPUcalculation
{
public:
    GPUcalculation();
    ~GPUcalculation();

    void getMatrixMultiply(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B);
    Eigen::MatrixXd getMatrixTranspose(const Eigen::MatrixXd &A);

private:
    cublasHandle_t handle;

    void checkCuda(cudaError_t result, const char *msg);
    void checkCublas(cublasStatus_t result, const char *msg);
};

#endif // GPUCALCULATION_H