
class RandomGenerator
{
public:
    static double generateRandomNumber(double min_rnd, double max_rnd);
    static Matrixd generateRandomMatrix(double min_rnd, double max_rnd, size_t rows, size_t cols);

private:
    RandomGenerator() = default;
    ~RandomGenerator() = default;

    RandomGenerator(const RandomGenerator &) = delete;
    RandomGenerator &operator=(const RandomGenerator &) = delete;
};