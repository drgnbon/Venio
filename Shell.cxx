#include <Eigen\Core>
#include <utility>
#include <iostream>


class Array {
public:

    Eigen::ArrayXXd _array;

    Array(Eigen::ArrayXXd array) : _array(std::move(array)) {}

    Array(const Eigen::ArrayXd& array) : _array(array) {}

    Array(const Eigen::MatrixXd& matrix)
    {
        _array = matrix;
    }



    size_t getRows() const {
        return _array.rows();
    }

    size_t getCols() const {
        return _array.cols();
    }

    friend Array operator*(const Array& a1, const Array& a2) {
        if (a1.getRows() != a2.getRows() || a1.getCols() != a2.getCols()) {
            throw std::invalid_argument("Array multiplication error: dimensions do not match");
        }
        return Array(Eigen::ArrayXXd(a1._array * a2._array));
    }
};










class Matrix{
public:
    //private in future
    size_t cols,rows;
    Eigen::MatrixXd _matrix;



    //private in future


    //Constuctors

    Matrix(Array array)
    {
        _matrix = array._array;
    }



    Matrix(size_t cols,size_t rows)
    {
        _matrix = Eigen::MatrixXd(cols,rows);
    }
    Matrix(Eigen::MatrixXd matrix)
    {
        _matrix = std::move(matrix);
    }


    //Constuctors


    //Functions
    size_t getRows() const
    {
        return _matrix.rows();
    }
    size_t getCols() const
    {
        return _matrix.cols();
    }

    Matrix getTranspose(){
        return Matrix(Eigen::MatrixXd(_matrix.transpose()));
    }

    Array array()
    {
        return Array(_matrix);
    }




    //Functions



    //Operators

    //+
    friend Matrix operator+(const Matrix& m1, const Matrix& m2) {
        if (m1.getRows() != m2.getRows() || m1.getCols() != m2.getCols()) {
            throw std::invalid_argument("Matrix addition error (m1.getRows != m2.getRows || m1.getCols != m2.getCols)");
        }
        return Matrix(Eigen::MatrixXd(m1._matrix + m2._matrix));
    }
    //+

    //-
    friend Matrix operator-(const Matrix& m1, const Matrix& m2) {
        if (m1.getRows() != m2.getRows() || m1.getCols() != m2.getCols()) {
            throw std::invalid_argument("Matrix difference error (m1.getRows != m2.getRows || m1.getCols != m2.getCols)");
        }
        return Matrix(Eigen::MatrixXd(m1._matrix - m2._matrix));
    }
    //-


    //*
    friend Matrix operator*(const Matrix& m1, const Matrix& m2) {
        if (m1.getCols() != m2.getRows()) {
            throw std::invalid_argument("Matrix multiplication error (m1.getCols != m2.getRows)");
        }
        return Matrix(Eigen::MatrixXd(m1._matrix * m2._matrix));
    }
    //*
    //Operators


    //friend Eigen::MatrixXd operator*();
    //friend Eigen::MatrixXd operator-();


};





int main()
{
    Matrix m1 = Eigen::MatrixXd::Constant(2,2,1);

    Matrix m2 =  Eigen::MatrixXd::Constant(2,2,1);


    Matrix m3 = m1.array()*m2.array();


    std::cout << m1._matrix << "\n\n";

    std::cout << m2._matrix << "\n\n";

    std::cout << m3._matrix << "\n\n";







}