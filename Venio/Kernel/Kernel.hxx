#pragma once

#include "Config.hxx"
#include <vector>
#include <thread>

class Kernel
{
public:
    static Matrixd multiplyMatrices(const Matrixd &a, const Matrixd &b);
    static Matrixd divideMatrices(const Matrixd &a, const Matrixd &b); // not supported
    static Matrixd addMatrices(const Matrixd &a, const Matrixd &b);
    static Matrixd subtractMatrices(const Matrixd &a, const Matrixd &b);
    static Matrixd transposeMatrix(const Matrixd &a);
    static Matrixd elementwiseMultiplyMatrices(const Matrixd &a, const Matrixd &b);
    static Matrixd elementwiseProductMatrices(); // not supported
    static Matrixd applyFunctionToElements(Matrixd a, double (*foo)(double));
};