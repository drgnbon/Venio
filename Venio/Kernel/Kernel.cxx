#include "Kernel.hxx"

namespace Kernel {

    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns result of multiply of two matrices</returns>
    Matrixd multiply(const Matrixd& a, const Matrixd& b)
    {
        return a * b;
    }
    // Matrixd Kernel::divideMatrices(const Matrixd &a, const Matrixd &b); // not supported


    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns summ of two matrices</returns>
    Matrixd sum(const Matrixd& a, const Matrixd& b)
    {
        return a + b;
    }


    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns subtract of two matrices</returns>
    Matrixd sub(const Matrixd& a, const Matrixd& b)
    {
        return a - b;
    }

    /// <param name="a">Eigen::MatrixXd </param>
    /// <returns>Returns trasposed matrix </returns>
    Matrixd transpose(const Matrixd& a)
    {
        return a.transpose();
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



#ifdef CPU_OPTIMIZATION

    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns result of multiply of two matrices</returns>
    Matrixd multiply(const Matrixd& a, const Matrixd& b)
    {
        return a * b;
    }
    // Matrixd Kernel::divideMatrices(const Matrixd &a, const Matrixd &b); // not supported


    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns summ of two matrices</returns>
    Matrixd sum(const Matrixd& a, const Matrixd& b)
    {
        return a + b;
    }


    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns subtract of two matrices</returns>
    Matrixd sub(const Matrixd& a, const Matrixd& b)
    {
        return a - b;
    }

    /// <param name="a">Eigen::MatrixXd </param>
    /// <returns>Returns trasposed matrix </returns>
    Matrixd transpose(const Matrixd& a)
    {
        return a.transpose();
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

#endif

#ifdef GPU_OPTIMIZATION
    /// <param name="a"> - element 1</param>
/// <param name="b"> - element 2</param>
/// <returns>Returns result of multiply of two matrices</returns>
    Matrixd multiply(const Matrixd& a, const Matrixd& b)
    {
        return a * b;
    }
    // Matrixd Kernel::divideMatrices(const Matrixd &a, const Matrixd &b); // not supported


    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns summ of two matrices</returns>
    Matrixd sum(const Matrixd& a, const Matrixd& b)
    {
        return a + b;
    }


    /// <param name="a"> - element 1</param>
    /// <param name="b"> - element 2</param>
    /// <returns>Returns subtract of two matrices</returns>
    Matrixd sub(const Matrixd& a, const Matrixd& b)
    {
        return a - b;
    }

    /// <param name="a">Eigen::MatrixXd </param>
    /// <returns>Returns trasposed matrix </returns>
    Matrixd transpose(const Matrixd& a)
    {
        return a.transpose();
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

#endif
};


