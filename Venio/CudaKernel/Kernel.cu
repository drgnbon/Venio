#include "Kernel.hxx"

namespace Kernel {
    //GPU WARNINGS--------------------------------------------------------------------------------------------------
    inline void checkCuda(cudaError_t result, const char* msg)
    {
        if (result != cudaSuccess)
        {
            std::cerr << "CUDA Error: " << msg << " - " << cudaGetErrorString(result) << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    inline void checkCublas(cublasStatus_t result, const char* msg)
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
    //GPU WARNINGS--------------------------------------------------------------------------------------------------------

    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns result of multiply of two matrices</returns>
    Matrixd multiply(const Matrixd& A, const Matrixd& B)
    {
        typedef void* internal_handle;
        internal_handle handle;
        cublasHandle_t tHandle;
        checkCublas(cublasCreate(&tHandle), "Failed to create cuBLAS handle");
        handle = static_cast<internal_handle>(tHandle);

        int m = static_cast<int>(A.rows());
        int n = static_cast<int>(B.cols());
        int k = static_cast<int>(A.cols());
        Eigen::MatrixXd C(m, n);

        const double alpha = 1.0;
        const double beta = 0.0;

        size_t size_of_A = m * k * sizeof(double);
        size_t size_of_B = n * k * sizeof(double);
        size_t size_of_C = m * n * sizeof(double);

        double* p_A = nullptr, * p_B = nullptr, * p_C = nullptr;

        checkCuda(cudaMalloc((void**)&p_A, size_of_A), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&p_B, size_of_B), "Failed to allocate GPU memory for B");
        checkCuda(cudaMalloc((void**)&p_C, size_of_C), "Failed to allocate GPU memory for C");

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

        cublasDestroy(getCublasHandle(handle));

        return C;
    }
    /*Matrixd multiply(const Matrixd& A, const Matrixd& B)
    {
        return A * B;
    }*/

    /* /// <param name="a">Eigen::MatrixXd </param>
    /// <returns>Returns trasposed matrix </returns>
    Matrixd transpose(const Matrixd& A)
    {
        return A.transpose();
    }*/

    /// <param name="a">Eigen::MatrixXd </param>
    /// <returns>Returns trasposed matrix </returns>
    Matrixd transpose(const Matrixd& A)
    {
        typedef void* internal_handle;
        internal_handle handle;
        cublasHandle_t tHandle;
        checkCublas(cublasCreate(&tHandle), "Failed to create cuBLAS handle");
        handle = static_cast<internal_handle>(tHandle);

        int rows = static_cast<int>(A.rows());
        int cols = static_cast<int>(A.cols());
        Eigen::MatrixXd B(cols, rows);

        size_t size_of_A = rows * cols * sizeof(double);
        size_t size_of_B = rows * cols * sizeof(double);

        double* p_A = nullptr, * p_B = nullptr;

        checkCuda(cudaMalloc((void**)&p_A, size_of_A), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&p_B, size_of_B), "Failed to allocate GPU memory for B");

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

        cublasDestroy(getCublasHandle(handle));

        return B;
    }






    Arrayd Kernel::divideArrays(const Arrayd& a, const Arrayd& b)
    {
        return a / b;
    }


    Matrixd scalarDivide(const Matrixd& a, double s)
    {
        return a / s;
    }



    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns summ of two matrices</returns>
    Matrixd sum(const Matrixd& a, const Matrixd& b)
    {
        return a + b;
    }
    

    Arrayd sqr(const Arrayd& a) {
        return a.square();
    }
    Arrayd sqrt(const Arrayd& a) {
        return Eigen::sqrt(a);
    }
    Arrayd scalarAdd(const Arrayd& a, double b) {
        return a + b;
    }
    Arrayd multiplyArrays(const Arrayd& a, const Arrayd& b) {
        return a * b;
    }
    Matrixd scalarArrayMultiply(const Arrayd& a, double s) {
        return s * a;
    }
    Matrixd scalarArrayMultiply(double s, const Arrayd& a) {
        return s * a;
    }

    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns subtract of two matrices</returns>
    Matrixd sub(const Matrixd& a, const Matrixd& b)
    {
        return a - b;
    }



    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns result of elemental multiply of two matrices </returns>
    Matrixd eMultiply(const Matrixd& a, const Matrixd& b)
    {
        return a.array() * b.array();
    }
    // Matrixd Kernel::elementwiseProductMatrices(); // not supported


    /// <param name="a"> - matrix</param>
    /// <param name="b"> - scalar </param>
    /// <returns>Returns result of scalar multiply (matrix and scalar) </returns>
    Matrixd scalarMultiply(Matrixd a, double s)
    {
        return a * s;
    }
    Matrixd scalarMultiply(double s, Matrixd a) {
        return a * s;
    }
    double dot(Eigen::VectorXd a, Eigen::VectorXd b) {
        return a.dot(b);
    }

    /// <param name="a"> - matrix.array()</param>
    /// <param name="b"> - scalar </param>
    /// <returns>Returns result of scalar sum (matrix and scalar) </returns>
    Arrayd scalarSum(Arrayd a, double s)
    {
        return a.array() + s;
    }



    /// <param name="a"> - matrix</param>
    /// <param name="foo"> - function </param>
    /// <returns>Returns matrix that was applied with function (foo) </returns>
    Matrixd applyFunctionToElements(Matrixd a, double (*foo)(double))
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




};


