#include "Kernel.hxx"
#include "Kernel.hxx"
#include "Kernel.hxx"

namespace Kernel {
    //GPU WARNINGS--------------------------
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
    //GPU WARNINGS----------------------------

    //----------------------------------------
    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns result of multiply of two matrices</returns>
    Matrixd multMM(const Matrixd& A, const Matrixd& B)
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
    //----------------------------------------

    //----------------------------------------
    /// <param name="a">Eigen::MatrixXd </param>
    /// <returns>Returns trasposed matrix </returns>
    Matrixd transposeM(const Matrixd& A)
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
    //----------------------------------------

    //----------------------------------------
    Arrayd divAA(const Arrayd& a, const Arrayd& b)
    {
        return a / b;
    }
    //----------------------------------------


    //----------------------------------------
    Matrixd divMS(const Matrixd& a, double s)
    {
        return a / s;
    }
    Matrixd divMS(double s, const Matrixd& a)
    {
        return a / s;
    }
    //----------------------------------------


    //----------------------------------------
    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns summ of two matrices</returns>
    Matrixd sumMM(const Matrixd& a, const Matrixd& b) {
        if (a.rows() != b.rows() || a.cols() != b.cols()) {
            throw std::invalid_argument("Matrix dimensions must match");
        }

        int rows = static_cast<int>(a.rows());
        int cols = static_cast<int>(a.cols());
        int size = rows * cols;

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для матриц a, b и результирующей матрицы c
        double* d_a = nullptr;
        double* d_b = nullptr;
        double* d_c = nullptr;
        size_t bytes = size * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_a, bytes), "Failed to allocate GPU memory for a");
        checkCuda(cudaMalloc((void**)&d_b, bytes), "Failed to allocate GPU memory for b");
        checkCuda(cudaMalloc((void**)&d_c, bytes), "Failed to allocate GPU memory for c");

        // Копируем данные матриц a и b на GPU
        checkCuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy data to GPU for a");
        checkCuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy data to GPU for b");

        const double alpha = 1.0;
        const double beta = 1.0;

        // Выполняем операцию сложения матриц c = alpha * a + beta * b
        checkCublas(cublasDgeam(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rows, cols,
            &alpha,
            d_a, rows,
            &beta,
            d_b, rows,
            d_c, rows),
            "Failed to perform matrix addition");

        // Создаем результирующую матрицу на CPU
        Matrixd c(rows, cols);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(c.data(), d_c, bytes, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for c");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_a), "Failed to free GPU memory for a");
        checkCuda(cudaFree(d_b), "Failed to free GPU memory for b");
        checkCuda(cudaFree(d_c), "Failed to free GPU memory for c");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return c;
    }
    //----------------------------------------



    //----------------------------------------
    Matrixd transposeV(const Eigen::MatrixXd& A)
    {

        return A.transpose();

        typedef void* internal_handle;
        internal_handle handle;
        cublasHandle_t tHandle;
        checkCublas(cublasCreate(&tHandle), "Failed to create cuBLAS handle");
        handle = static_cast<internal_handle>(tHandle);

        int rows = static_cast<int>(A.rows());
        int cols = static_cast<int>(A.cols());

        // Если матрица не является вектором, выбрасываем исключение
        if (rows != 1 && cols != 1) {
            throw std::invalid_argument("Input is not a vector");
        }

        // Определяем размеры для транспонированного вектора
        int transposedRows = cols;
        int transposedCols = rows;

        Eigen::MatrixXd B(transposedRows, transposedCols);

        size_t size_of_A = rows * cols * sizeof(double);
        size_t size_of_B = transposedRows * transposedCols * sizeof(double);

        double* p_A = nullptr;
        double* p_B = nullptr;

        checkCuda(cudaMalloc((void**)&p_A, size_of_A), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&p_B, size_of_B), "Failed to allocate GPU memory for B");

        checkCuda(cudaMemcpy(p_A, A.data(), size_of_A, cudaMemcpyHostToDevice), "Failed to copy data to GPU");

        const double alpha = 1.0;
        const double beta = 0.0;

        // Используем cuBLAS для выполнения операции транспонирования
        checkCublas(cublasDgeam(tHandle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            transposedRows, transposedCols,
            &alpha,
            p_A, rows,
            &beta,
            p_A, rows,
            p_B, transposedRows),
            "Failed to perform vector transpose");

        checkCuda(cudaMemcpy(B.data(), p_B, size_of_B, cudaMemcpyDeviceToHost), "Failed to copy data from GPU");

        checkCuda(cudaFree(p_A), "Failed to free GPU memory for A");
        checkCuda(cudaFree(p_B), "Failed to free GPU memory for B");

        cublasDestroy(tHandle);

        return B;
    }
    //----------------------------------------


    //----------------------------------------
    Matrixd multVV(const Vectord& a, const Vectord& b)
    {
        return a*b;
    }
    //----------------------------------------

    //----------------------------------------
    Arrayd sqrA(const Arrayd& a) {
        return a.square();
    }
    //----------------------------------------

    //----------------------------------------
    Arrayd sqrtA(const Arrayd& a) {
        return Eigen::sqrt(a);
    }
    //----------------------------------------

    //----------------------------------------
    Arrayd sumAS(const Arrayd& a, double b) {
        return a + b;
    }
    Arrayd sumAS(double b, const Arrayd& a) {
        return a + b;
    }
    //----------------------------------------

    //----------------------------------------
    Arrayd multAA(const Arrayd& a, const Arrayd& b) {
        return a * b;
    }
    //----------------------------------------

    //----------------------------------------
    Arrayd multAS(const Arrayd& a, double s) {
        return s * a;
    }
    Arrayd multAS( double s, const Arrayd& a) {
        return s * a;
    }
    //----------------------------------------


    //----------------------------------------
    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns subtract of two matrices</returns>
    Matrixd subMM(const Matrixd& A, const Matrixd& B) 
    {

        if (A.rows() != B.rows() || A.cols() != B.cols()) {
            throw std::invalid_argument("Matrices dimensions must match");
        }

        int rows = static_cast<int>(A.rows());
        int cols = static_cast<int>(A.cols());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для матриц A, B и C
        double* d_A = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;
        size_t size = rows * cols * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_A, size), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&d_B, size), "Failed to allocate GPU memory for B");
        checkCuda(cudaMalloc((void**)&d_C, size), "Failed to allocate GPU memory for C");

        // Копируем данные матриц A и B на GPU
        checkCuda(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice), "Failed to copy data to GPU for A");
        checkCuda(cudaMemcpy(d_B, B.data(), size, cudaMemcpyHostToDevice), "Failed to copy data to GPU for B");

        const double alpha = 1.0;
        const double beta = -1.0;

        // Выполняем операцию вычитания d_C = alpha * d_A + beta * d_B (т.е. d_C = d_A - d_B)
        checkCublas(cublasDgeam(handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            rows, cols,
            &alpha,
            d_A, rows,
            &beta,
            d_B, rows,
            d_C, rows),
            "Failed to perform matrix subtraction");

        // Создаем результирующую матрицу на CPU
        Matrixd C(rows, cols);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for C");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_A), "Failed to free GPU memory for A");
        checkCuda(cudaFree(d_B), "Failed to free GPU memory for B");
        checkCuda(cudaFree(d_C), "Failed to free GPU memory for C");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return C;
    }
    //----------------------------------------

    //----------------------------------------
    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns result of elemental multiply of two matrices </returns>
    Matrixd emultMM(const Matrixd& a, const Matrixd& b)
    {
        return a.array() * b.array();
    }
    //----------------------------------------

    //----------------------------------------
    /// <param name="a"> - matrix</param>
    /// <param name="b"> - scalar </param>
    /// <returns>Returns result of scalar multiply (matrix and scalar) </returns>
    Matrixd multMS(const Matrixd& A, double S) {
        int rows = static_cast<int>(A.rows());
        int cols = static_cast<int>(A.cols());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для матрицы A и результирующей матрицы C
        double* d_A = nullptr;
        double* d_C = nullptr;
        size_t size = rows * cols * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_A, size), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&d_C, size), "Failed to allocate GPU memory for C");

        // Копируем данные матрицы A на GPU
        checkCuda(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice), "Failed to copy data to GPU for A");

        // Выполняем операцию умножения на скаляр d_C = S * d_A
        checkCublas(cublasDscal(handle, rows * cols, &S, d_A, 1), "Failed to perform matrix scaling");

        // Копируем результат из d_A в d_C (если необходимо)
        checkCuda(cudaMemcpy(d_C, d_A, size, cudaMemcpyDeviceToDevice), "Failed to copy data from d_A to d_C");

        // Создаем результирующую матрицу на CPU
        Matrixd C(rows, cols);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for C");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_A), "Failed to free GPU memory for A");
        checkCuda(cudaFree(d_C), "Failed to free GPU memory for C");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return C;
    }
    Matrixd multMS(double S, const Matrixd& A)
    {
        int rows = static_cast<int>(A.rows());
        int cols = static_cast<int>(A.cols());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для матрицы A и результирующей матрицы C
        double* d_A = nullptr;
        double* d_C = nullptr;
        size_t size = rows * cols * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_A, size), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&d_C, size), "Failed to allocate GPU memory for C");

        // Копируем данные матрицы A на GPU
        checkCuda(cudaMemcpy(d_A, A.data(), size, cudaMemcpyHostToDevice), "Failed to copy data to GPU for A");

        // Выполняем операцию умножения на скаляр d_C = S * d_A
        checkCublas(cublasDscal(handle, rows * cols, &S, d_A, 1), "Failed to perform matrix scaling");

        // Копируем результат из d_A в d_C (если необходимо)
        checkCuda(cudaMemcpy(d_C, d_A, size, cudaMemcpyDeviceToDevice), "Failed to copy data from d_A to d_C");

        // Создаем результирующую матрицу на CPU
        Matrixd C(rows, cols);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(C.data(), d_C, size, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for C");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_A), "Failed to free GPU memory for A");
        checkCuda(cudaFree(d_C), "Failed to free GPU memory for C");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return C;
    }
    //----------------------------------------

    //----------------------------------------
    double dotVV(const Vectord& a, const Vectord& b) {
        if (a.size() != b.size()) {
            throw std::invalid_argument("Vectors must be the same size");
        }

        int size = static_cast<int>(a.size());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для векторов a и b
        double* d_a = nullptr;
        double* d_b = nullptr;
        size_t bytes = size * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_a, bytes), "Failed to allocate GPU memory for a");
        checkCuda(cudaMalloc((void**)&d_b, bytes), "Failed to allocate GPU memory for b");

        // Копируем данные векторов a и b на GPU
        checkCuda(cudaMemcpy(d_a, a.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy data to GPU for a");
        checkCuda(cudaMemcpy(d_b, b.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy data to GPU for b");

        double result = 0.0;

        // Выполняем операцию скалярного произведения
        checkCublas(cublasDdot(handle, size, d_a, 1, d_b, 1, &result), "Failed to perform dot product");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_a), "Failed to free GPU memory for a");
        checkCuda(cudaFree(d_b), "Failed to free GPU memory for b");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return result;
    }
    //----------------------------------------

    //----------------------------------------
    /// <param name="a"> - matrix.array()</param>
    /// <param name="b"> - scalar </param>
    /// <returns>Returns result of scalar sum (matrix and scalar) </returns>
    Arrayd sumAS(Arrayd a, double s)
    {
        return a.array() + s;
    }
    Arrayd sumAS(double s, Arrayd a)
    {
        return a.array() + s;
    }
    //----------------------------------------

    //----------------------------------------
    /// <param name="a"> - matrix</param>
    /// <param name="foo"> - function </param>
    /// <returns>Returns matrix that was applied with function (foo) </returns>
    Matrixd applyFuncMF(Matrixd a, double (*foo)(double))
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
    Matrixd applyFuncMF(double (*foo)(double), Matrixd a)
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
    //----------------------------------------

    //----------------------------------------
    Arrayd sumAA(const Arrayd& a, const Arrayd& b) 
    {
        return a + b;
    }
    //----------------------------------------

    //----------------------------------------
    Vectord multMV(const Matrixd& A, const Vectord& B) 
    {
        if (A.cols() != B.size()) {
            throw std::invalid_argument("Matrix columns must match vector size");
        }

        int rows = static_cast<int>(A.rows());
        int cols = static_cast<int>(A.cols());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для матрицы A и вектора B и результирующего вектора C
        double* d_A = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;
        size_t size_A = rows * cols * sizeof(double);
        size_t size_B = cols * sizeof(double);
        size_t size_C = rows * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_A, size_A), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&d_B, size_B), "Failed to allocate GPU memory for B");
        checkCuda(cudaMalloc((void**)&d_C, size_C), "Failed to allocate GPU memory for C");

        // Копируем данные матрицы A и вектора B на GPU
        checkCuda(cudaMemcpy(d_A, A.data(), size_A, cudaMemcpyHostToDevice), "Failed to copy data to GPU for A");
        checkCuda(cudaMemcpy(d_B, B.data(), size_B, cudaMemcpyHostToDevice), "Failed to copy data to GPU for B");

        const double alpha = 1.0;
        const double beta = 0.0;

        // Выполняем операцию умножения матрицы на вектор d_C = alpha * d_A * d_B + beta * d_C
        checkCublas(cublasDgemv(handle,
            CUBLAS_OP_N,
            rows, cols,
            &alpha,
            d_A, rows,
            d_B, 1,
            &beta,
            d_C, 1),
            "Failed to perform matrix-vector multiplication");

        // Создаем результирующий вектор на CPU
        Vectord C(rows);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(C.data(), d_C, size_C, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for C");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_A), "Failed to free GPU memory for A");
        checkCuda(cudaFree(d_B), "Failed to free GPU memory for B");
        checkCuda(cudaFree(d_C), "Failed to free GPU memory for C");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return C;
    }


    Vectord multVM(const Vectord& a, const Matrixd& b) 
    {
        if (a.size() != b.rows()) {
            throw std::invalid_argument("Vector size must match the number of rows in the matrix");
        }

        int rows = static_cast<int>(b.rows());
        int cols = static_cast<int>(b.cols());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для вектора a, матрицы b и результирующего вектора c
        double* d_a = nullptr;
        double* d_b = nullptr;
        double* d_c = nullptr;
        size_t size_a = rows * sizeof(double);
        size_t size_b = rows * cols * sizeof(double);
        size_t size_c = cols * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_a, size_a), "Failed to allocate GPU memory for a");
        checkCuda(cudaMalloc((void**)&d_b, size_b), "Failed to allocate GPU memory for b");
        checkCuda(cudaMalloc((void**)&d_c, size_c), "Failed to allocate GPU memory for c");

        // Копируем данные вектора a и матрицы b на GPU
        checkCuda(cudaMemcpy(d_a, a.data(), size_a, cudaMemcpyHostToDevice), "Failed to copy data to GPU for a");
        checkCuda(cudaMemcpy(d_b, b.data(), size_b, cudaMemcpyHostToDevice), "Failed to copy data to GPU for b");

        const double alpha = 1.0;
        const double beta = 0.0;

        // Выполняем операцию умножения вектора на матрицу d_c = alpha * d_b^T * d_a + beta * d_c
        // Используем CUBLAS_OP_T для транспонирования матрицы b
        checkCublas(cublasDgemv(handle,
            CUBLAS_OP_T,
            rows, cols,
            &alpha,
            d_b, rows,
            d_a, 1,
            &beta,
            d_c, 1),
            "Failed to perform vector-matrix multiplication");

        // Создаем результирующий вектор на CPU
        Vectord c(cols);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(c.data(), d_c, size_c, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for c");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_a), "Failed to free GPU memory for a");
        checkCuda(cudaFree(d_b), "Failed to free GPU memory for b");
        checkCuda(cudaFree(d_c), "Failed to free GPU memory for c");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return c;
    }
    //----------------------------------------

    //----------------------------------------
    Vectord multVS(const Vectord& A, double S) 
    {
        int size = static_cast<int>(A.size());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для вектора A и результирующего вектора C
        double* d_A = nullptr;
        size_t bytes = size * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_A, bytes), "Failed to allocate GPU memory for A");

        // Копируем данные вектора A на GPU
        checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy data to GPU for A");

        // Выполняем операцию умножения на скаляр d_A = S * d_A
        checkCublas(cublasDscal(handle, size, &S, d_A, 1), "Failed to perform vector scaling");

        // Создаем результирующий вектор на CPU
        Vectord C(size);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(C.data(), d_A, bytes, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for C");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_A), "Failed to free GPU memory for A");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return C;
    }
    Vectord multVS(double S, const Vectord& A) 
    {
        int size = static_cast<int>(A.size());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для вектора A и результирующего вектора C
        double* d_A = nullptr;
        size_t bytes = size * sizeof(double);

        checkCuda(cudaMalloc((void**)&d_A, bytes), "Failed to allocate GPU memory for A");

        // Копируем данные вектора A на GPU
        checkCuda(cudaMemcpy(d_A, A.data(), bytes, cudaMemcpyHostToDevice), "Failed to copy data to GPU for A");

        // Выполняем операцию умножения на скаляр d_A = S * d_A
        checkCublas(cublasDscal(handle, size, &S, d_A, 1), "Failed to perform vector scaling");

        // Создаем результирующий вектор на CPU
        Vectord C(size);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(C.data(), d_A, bytes, cudaMemcpyDeviceToHost), "Failed to copy data from GPU for C");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_A), "Failed to free GPU memory for A");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return C;
    }
    //----------------------------------------
 
    //----------------------------------------
    Vectord subVV(const Vectord& A, const Vectord& B) {
        if (A.size() != B.size()) {
            throw std::invalid_argument("Vectors must have the same size");
        }

        int size = static_cast<int>(A.size());

        // Создаем и инициализируем cuBLAS handle
        cublasHandle_t handle;
        checkCublas(cublasCreate(&handle), "Failed to create cuBLAS handle");

        // Выделяем память на GPU для векторов A, B и C
        double* d_A = nullptr;
        double* d_B = nullptr;
        double* d_C = nullptr;

        checkCuda(cudaMalloc((void**)&d_A, size * sizeof(double)), "Failed to allocate GPU memory for A");
        checkCuda(cudaMalloc((void**)&d_B, size * sizeof(double)), "Failed to allocate GPU memory for B");
        checkCuda(cudaMalloc((void**)&d_C, size * sizeof(double)), "Failed to allocate GPU memory for C");

        // Копируем данные векторов A и B на GPU
        checkCuda(cudaMemcpy(d_A, A.data(), size * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy data to GPU for A");
        checkCuda(cudaMemcpy(d_B, B.data(), size * sizeof(double), cudaMemcpyHostToDevice), "Failed to copy data to GPU for B");

        const double alpha = 1.0;
        const double beta = -1.0;

        // Выполняем операцию вычитания d_C = alpha * d_A + beta * d_B (т.е. d_C = d_A - d_B)
        checkCublas(cublasDaxpy(handle, size, &beta, d_B, 1, d_A, 1), "Failed to perform vector subtraction");
        checkCuda(cudaMemcpy(d_C, d_A, size * sizeof(double), cudaMemcpyDeviceToDevice), "Failed to copy result to d_C");

        // Создаем результирующий вектор на CPU
        Vectord C(size);

        // Копируем результат обратно на CPU
        checkCuda(cudaMemcpy(C.data(), d_C, size * sizeof(double), cudaMemcpyDeviceToHost), "Failed to copy data from GPU for C");

        // Освобождаем память на GPU
        checkCuda(cudaFree(d_A), "Failed to free GPU memory for A");
        checkCuda(cudaFree(d_B), "Failed to free GPU memory for B");
        checkCuda(cudaFree(d_C), "Failed to free GPU memory for C");

        // Уничтожаем cuBLAS handle
        checkCublas(cublasDestroy(handle), "Failed to destroy cuBLAS handle");

        return C;
    }
    //----------------------------------------
};


