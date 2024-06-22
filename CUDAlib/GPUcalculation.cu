#include "GPUcalculation.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>

inline void checkCuda(cudaError_t result, const char *msg)
{
    if (result != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline void checkCublas(cublasStatus_t result, const char *msg)
{
    if (result != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS Error: " << msg << std::endl;
        exit(EXIT_FAILURE);
    }
}

inline cublasHandle_t getCublasHandle(void* handle) noexcept
{
    return reinterpret_cast<cublasHandle_t>(handle);
}

GPUcalculation::GPUcalculation()
{
    cublasHandle_t tHandle;
    checkCublas(cublasCreate(&tHandle), "Failed to create cuBLAS handle");
    handle = static_cast<internal_handle>(tHandle);
}

GPUcalculation::~GPUcalculation()
{
    cublasDestroy(getCublasHandle(handle));
}

void GPUcalculation::getMatrixMultiply(const Eigen::MatrixXd &A, const Eigen::MatrixXd &B)
{
    int m = static_cast<int>(A.rows());
    int n = static_cast<int>(B.cols());
    int k = static_cast<int>(A.cols());
    Eigen::MatrixXd C(m, n);

    const double alpha = 1.0;
    const double beta = 0.0;

    size_t size_of_A = m * k * sizeof(double);
    size_t size_of_B = n * k * sizeof(double);
    size_t size_of_C = m * n * sizeof(double);

    double *p_A = nullptr, *p_B = nullptr, *p_C = nullptr;

    checkCuda(cudaMalloc((void **)&p_A, size_of_A), "Failed to allocate GPU memory for A");
    checkCuda(cudaMalloc((void **)&p_B, size_of_B), "Failed to allocate GPU memory for B");
    checkCuda(cudaMalloc((void **)&p_C, size_of_C), "Failed to allocate GPU memory for C");

    checkCuda(cudaMemcpy(p_A, A.data(), size_of_A, cudaMemcpyHostToDevice), "Failed to copy data to GPU");
    checkCuda(cudaMemcpy(p_B, B.data(), size_of_B, cudaMemcpyHostToDevice), "Failed to copy data to GPU");

    checkCublas(cublasDgemm(getCublasHandle(handle),
                            CUBLAS_OP_N, CUBLAS_OP_N,
                            m, n, k,
                            &alpha,
                            p_A, m,
                            p_B, k,
                            &beta,
                            p_C, m),
                "Failed to perform DGEMM");

    checkCuda(cudaMemcpy(C.data(), p_C, size_of_C, cudaMemcpyDeviceToHost), "Failed to copy data from GPU");

    checkCuda(cudaFree(p_A), "Failed to free GPU memory for A");
    checkCuda(cudaFree(p_B), "Failed to free GPU memory for B");
    checkCuda(cudaFree(p_C), "Failed to free GPU memory for C");
}

Eigen::MatrixXd GPUcalculation::getMatrixTranspose(const Eigen::MatrixXd &A)
{
    int rows = static_cast<int>(A.rows());
    int cols = static_cast<int>(A.cols());
    Eigen::MatrixXd B(cols, rows);

    size_t size_of_A = rows * cols * sizeof(double);
    size_t size_of_B = rows * cols * sizeof(double);

    double *p_A = nullptr, *p_B = nullptr;

    checkCuda(cudaMalloc((void **)&p_A, size_of_A), "Failed to allocate GPU memory for A");
    checkCuda(cudaMalloc((void **)&p_B, size_of_B), "Failed to allocate GPU memory for B");

    checkCuda(cudaMemcpy(p_A, A.data(), size_of_A, cudaMemcpyHostToDevice), "Failed to copy data to GPU");

    const double alpha = 1.0;
    const double beta = 0.0;

    checkCublas(cublasDgeam(getCublasHandle(handle),
                            CUBLAS_OP_T, CUBLAS_OP_N,
                            cols, rows,
                            &alpha,
                            p_A, rows,
                            &beta,
                            p_A, cols,
                            p_B, cols),
                "Failed to perform matrix transpose");

    checkCuda(cudaMemcpy(B.data(), p_B, size_of_B, cudaMemcpyDeviceToHost), "Failed to copy data from GPU");

    checkCuda(cudaFree(p_A), "Failed to free GPU memory for A");
    checkCuda(cudaFree(p_B), "Failed to free GPU memory for B");

    return B;
}
