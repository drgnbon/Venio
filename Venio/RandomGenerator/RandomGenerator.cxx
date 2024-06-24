#include "RandomGenerator.hxx"

double RandomGenerator::generateRandomNumber(double min_rnd, double max_rnd)
{
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_real_distribution<double> rng_coin(min_rnd, max_rnd);
    return rng_coin(gen);
}

Matrixd RandomGenerator::generateRandomMatrix(double min_rnd, double max_rnd, size_t rows, size_t cols)
{
    min_rnd = -0.999;
    max_rnd = 0.999;
    std::random_device random_device;
    std::mt19937 gen(random_device());
    std::uniform_real_distribution<double> rng_coin(min_rnd, max_rnd);
    Matrixd matrix(rows, cols);
    for (long long i = 0; i < rows; ++i)
    {
        for (long long j = 0; j < cols; ++j)
        {
            matrix(i, j) = rng_coin(gen);
        }
    }
    return matrix;
}
