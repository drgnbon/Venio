#include <Eigen\Core>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
#include <fstream>
#include <random>
#include <filesystem>

using namespace Eigen;
typedef Eigen::MatrixXd Matrixd;


//Advanced generator------------------------------------------------------------------------------------------------------
class RandomGenerator {
public:
    static double generateRandomNumber(double min_rnd, double max_rnd) {
        std::random_device random_device;
        std::mt19937 gen(random_device());
        std::uniform_real_distribution<double> rng_coin(min_rnd, max_rnd);
        return rng_coin(gen);
    }

    static Matrixd generateRandomMatrix(double min_rnd, double max_rnd, size_t rows, size_t cols) {
        std::random_device random_device;
        std::mt19937 gen(random_device());
        std::uniform_real_distribution<double> rng_coin(min_rnd, max_rnd);
        Matrixd matrix(rows, cols);
        for (long long i = 0; i < rows; ++i) {
            for (long long j = 0; j < cols; ++j) {
                matrix(i, j) = rng_coin(gen);
            }
        }
        return matrix;
    }

private:
    RandomGenerator() = default;
    ~RandomGenerator() = default;

    RandomGenerator(const RandomGenerator&) = delete;
    RandomGenerator& operator=(const RandomGenerator&) = delete;
};
//Advanced generator------------------------------------------------------------------------------------------------------

// Exception logger------------------------------------------------------------------------------------------------------

class ErrorLogger
{
public:
    static ErrorLogger &getInstance()
    {
        static ErrorLogger instance;
        return instance;
    }

    void logError(const std::string &message)
    {
        std::cerr << "Error: " << message << std::endl;
        system("pause");
        exit(0);
    }

private:
    ErrorLogger() = default;
    ~ErrorLogger() = default;

    ErrorLogger(const ErrorLogger &) = delete;
    ErrorLogger &operator=(const ErrorLogger &) = delete;
};

// Exception logger-----------------------------------------------------------------------------------------------------

// ActivationFunctions---------------------------------------------------------------------------------------------------
class ActivationFunction
{
public:
    ActivationFunction() = default;
    ~ActivationFunction() = default;

    virtual double toActivateValue(double x) = 0;
    virtual double toDerivateValue(double x) = 0;

    Matrixd toActivateMatrix(Matrixd matrix)
    {
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                matrix(i, j) = toActivateValue(matrix(i, j));
            }
        }
        return matrix;
    }
    Matrixd toDerivateMatrix(Matrixd matrix)
    {
        for (int i = 0; i < matrix.rows(); ++i)
        {
            for (int j = 0; j < matrix.cols(); ++j)
            {
                matrix(i, j) = toDerivateValue(matrix(i, j));
            }
        }
        return matrix;
    }
};
class LogisticFunction : public ActivationFunction
{
public:
    double toActivateValue(double x) override
    {
        return 1.f / (1.f + exp(-x));
    }

    double toDerivateValue(double x) override
    {
        double activatedValue = toActivateValue(x);
        return activatedValue * (1.f - activatedValue);
    }
};
class LinearFunction : public ActivationFunction
{
public:
    double toActivateValue(double x) override
    {
        return x;
    }

    double toDerivateValue(double x) override
    {
        return 1.0;
    }
};
class SoftSignFunction : public ActivationFunction
{
public:
    double toActivateValue(double x) override
    {
        return x / (1.0 + fabs(x));  // Используйте std::abs
    }

    double toDerivateValue(double x) override
    {
        return 1.0 / std::pow(1.0 + fabs(x), 2);  // Используйте std::pow и std::abs
    }
};
class SincFunction : public ActivationFunction
{
public:
    double toActivateValue(double x) override
    {
        if (fabs(x) < 1e-8) return 1.0;
        return sin(x) / x;
    }

    double toDerivateValue(double x) override
    {
        if (fabs(x) < 1e-8) return 0.0;
        return (cos(x) / x) - (sin(x) / (x * x));
    }
};
class GhFunction : public ActivationFunction
{
public:
    double toActivateValue(double x) override
    {
        return exp(-(x * x));
    }

    double toDerivateValue(double x) override
    {
        return -2 * x * exp(-(x * x));
    }
};
// ActivationFunctions---------------------------------------------------------------------------------------------------

// LossFunctions---------------------------------------------------------------------------------------------------------
class LossFunction
{
public:
    virtual double getMediumLoss(const Matrixd& activeValue, const Matrixd& rightAnswer) = 0;
    virtual Matrixd getDerivationLoss(const Matrixd& activeValue, const Matrixd& rightAnswer) = 0;
};
class SquareErrorFunction : public LossFunction
{
public:
    double getMediumLoss(const Matrixd& activeValue, const Matrixd& rightAnswer) override
    {
        double squareError = (activeValue - rightAnswer).squaredNorm();
        return squareError / static_cast<double>(activeValue.size());
    }

    Matrixd getDerivationLoss(const Matrixd& activeValue, const Matrixd& rightAnswer) override
    {
        return 2.0 * (activeValue - rightAnswer);
    }
};
// LossFunctions---------------------------------------------------------------------------------------------------------

// Layers----------------------------------------------------------------------------------------------------------------
class Layer
{
protected:
    Matrixd _bias, _bias_gradient;
    Matrixd _values;
    Matrixd _active_values;
    Matrixd _derivation_neurons;
    Matrixd _weights, _weights_gradient;
    ActivationFunction *_activation_function;
    int _layer_size;

public:

    Layer(int layer_size, ActivationFunction *activation_function)
    {
        _layer_size = layer_size;
        _activation_function = activation_function;
    }
    virtual ~Layer() {}
    void buildLayer(int output_size_of_last_layer)
    {
        // Генерация случайных весов для связей между слоями
        _weights = RandomGenerator::generateRandomMatrix(0.001, 0.999, output_size_of_last_layer, _layer_size);

        // Инициализация градиентов весов нулями
        _weights_gradient = Matrixd::Zero(output_size_of_last_layer, _layer_size);

        // Инициализация активных значений слоя нулями
        _active_values = Matrixd::Zero(1, _layer_size);

        // Инициализация значений слоя нулями
        _values = Matrixd::Zero(1, _layer_size);

        // Генерация случайных значений для смещений (bias)
        _bias = RandomGenerator::generateRandomMatrix(0.001, 0.999, 1, _layer_size);

        // Инициализация градиентов смещений нулями
        _bias_gradient = Matrixd::Zero(1, _layer_size);

        // Инициализация производных нейронов нулями
        _derivation_neurons = Matrixd::Zero(1, _layer_size);
    }
    void buildFirstLayer(){
        // Инициализация активных значений слоя нулями
        _active_values = Matrixd::Zero(1, _layer_size);

        // Инициализация значений слоя нулями
        _values = Matrixd::Zero(1, _layer_size);

        // Генерация случайных значений для смещений (bias)
        _bias = RandomGenerator::generateRandomMatrix(0.001, 0.999, 1, _layer_size);

        // Инициализация градиентов смещений нулями
        _bias_gradient = Matrixd::Zero(1, _layer_size);

        // Инициализация производных нейронов нулями
        _derivation_neurons = Matrixd::Zero(1, _layer_size);
    }

    void activateLayer()
    {
        _active_values = _activation_function->toActivateMatrix(_values);
    }
    virtual void propogateLayer(Matrixd last_layer_output) = 0;
    virtual void backPropogateLayer(Matrixd next_layer_derivation, Matrixd next_layer_values, Matrixd next_layer_weights, Matrixd last_active_values) = 0;

    // getters & setters for class Layer---------------------------------------------------------------------

    void setLayerDerivation(Matrixd new_derivation_neurons_matrix)
    {
        if (!(new_derivation_neurons_matrix.cols() == _derivation_neurons.cols()
           && new_derivation_neurons_matrix.rows() == _derivation_neurons.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerDerivation (Not equal size of matrix) - class Layer \n");
        }

        _derivation_neurons = std::move(new_derivation_neurons_matrix);
    }
    void setLayerBias(Matrixd new_bias_matrix)
    {
        if (!(new_bias_matrix.cols() == _bias.cols()
           && new_bias_matrix.rows() == _bias.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerBias (Not equal size of matrix) - class Layer \n");
        }

        _bias = std::move(new_bias_matrix);
    }
    void setLayerBiasGradient(Matrixd new_bias_gradient_matrix)
    {
        if (!(new_bias_gradient_matrix.cols() == _bias_gradient.cols()
           && new_bias_gradient_matrix.rows() == _bias_gradient.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerBiasGradient (Not equal size of matrix) - class Layer \n");
        }
        _bias_gradient = std::move(new_bias_gradient_matrix);
    }
    void setLayerValues(Matrixd new_values_matrix)
    {
        if (!(new_values_matrix.cols() == _values.cols()
           && new_values_matrix.rows() == _values.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerValues (Not equal size of matrix) - class Layer \n");
        }

        _values = std::move(new_values_matrix);
    }
    void setLayerActiveValues(Matrixd new_active_values_matrix)
    {
        if (!(new_active_values_matrix.cols() == _active_values.cols()
           && new_active_values_matrix.rows() == _active_values.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerActiveValues (Not equal size of matrix) - class Layer \n");
        }

        _active_values = std::move(new_active_values_matrix);
    }
    void setLayerWeights(Matrixd new_weights_matrix)
    {
        if (!(new_weights_matrix.cols() == _weights.cols()
           && new_weights_matrix.rows() == _weights.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerWeights (Not equal size of matrix) - class Layer \n");
        }

        _weights = std::move(new_weights_matrix);
    }
    void setLayerWeightsGradient(Matrixd new_weights_gradient_matrix)
    {

        if (!(new_weights_gradient_matrix.cols() == _weights_gradient.cols()
           && new_weights_gradient_matrix.rows() == _weights_gradient.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerWeightsGradient (Not equal size of matrix) - class Layer \n");
        }

        _weights_gradient = std::move(new_weights_gradient_matrix);
    }
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    void setLayerActivationFunction(ActivationFunction *new_activation_function)
    {
        // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
        try{
            _activation_function = new_activation_function;
        }catch(std::exception& e){
            std::cerr << "Activation function error: " << e.what() << "\n";
        }
        // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING

    }

    // getters--------------------------------------------------------

    Matrixd getLayerBias()
    {
        return _bias;
    }
    Matrixd getLayerValues()
    {
        return _values;
    }
    Matrixd getLayerActiveValues()
    {
        return _active_values;
    }
    Matrixd getLayerWeights()
    {
        return _weights;
    }
    Matrixd getLayerBiasGradient()
    {
        return _bias_gradient;
    }
    Matrixd getLayerWeightsGradient()
    {
        return _weights_gradient;
    }
    //Rename or function name or name of value
    Matrixd getLayerDerivation()
    {
        return _derivation_neurons;
    }
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    ActivationFunction* getLayerActivationFunction()
    {
        // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
        return _activation_function;
        // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    }
    size_t getLayerSize() const
    {
        return _layer_size;
    }
    // getters & setters for class Layer---------------------------------------------------------------------
};

// work in progress--------------
class SequentialLayer : public Layer
{
public:
    SequentialLayer(int layer_size, ActivationFunction *activation_function)
        : Layer(layer_size, activation_function) {}

    void propogateLayer(Matrixd last_layer_output) override
    {
        _values = last_layer_output * _weights + _bias;
        activateLayer();
    }
    void backPropogateLayer(Matrixd next_layer_derivation, Matrixd next_layer_values, Matrixd next_layer_weights, Matrixd last_active_values) override
    {
        //Do work---------------------------------------------------------------------
    }
};

// work in progress--------------
/*
class ConvolutionLayer : Layer
{
public:
    ConvolutionLayer(int layer_size, ActivationFunction *activation_function)
        : Layer(layer_size, activation_function) {}

    void propogateLayer(Matrixd last_layer_output) override
    {
        // Work in progress--------------------------
    }
    void backPropogateLayer(Matrixd next_layer_derivation, Matrixd next_layer_values, Matrixd next_layer_weights, Matrixd last_active_values) override
    {
        // Work in progress--------------------------
    }
};*/

// Layers----------------------------------------------------------------------------------------------------------------


class Model
{
private:
    LossFunction *_loss_function;
    Matrixd _input, _output;
    std::vector<std::shared_ptr<Layer>> _layers;

public:
    Model(LossFunction *loss_function, const std::vector<std::shared_ptr<Layer>> &layers)
    {

        for (const auto i : layers)
        {
            addLayer(i);
        }

        _input = Matrixd::Zero(1, _layers[0]->getLayerSize());
        _output = Matrixd::Zero(1, _layers[_layers.size() - 1]->getLayerSize());
        _loss_function = loss_function;
    }
    ~Model()
    {
        _layers.clear();
    }

    void addLayer(std::shared_ptr<Layer> layer)
    {

        if(_layers.size() == 0){
            _layers.push_back(std::move(layer));
            _layers[0]->buildFirstLayer();
            return;
        }
        _layers.push_back(std::move(layer));
        _layers[_layers.size() - 1]->buildLayer(_layers[_layers.size() - 2]->getLayerSize());
    }

    void Log()
    {
        std::cout << "Input size : " << _input.cols() << ", Input: " << _input << "\n";

        printf("Number of layers: %zu \n", _layers.size());

        std::cout << "Layers: \n";

        int k = 1;

        for (auto i : _layers)
        {
            printf("\tLayer num: %d,Layer size: %d,Layer active values: ", k++, i->getLayerSize());
            std::cout << i->getLayerActiveValues() << "\n";
            std::cout << "\tLayer derivations: " << i->getLayerDerivation() << "\n\n";
        }
        std::cout << "Output size : " << _output.cols() << ", Output: " << _output << "\n";
    }

    void forwardPropogation()
    {
        for (int i = 1; i < _layers.size(); ++i)
        {
            _layers[i]->propogateLayer(_layers[i - 1]->getLayerActiveValues());
        }

        // Add activation to output (maybe softmax)-----------------
        _output = _layers[_layers.size() - 1]->getLayerActiveValues();
    }
    void backPropogation(Matrixd right_answer)
    {

        //Do work------------------------------------------------------------------

    }

    // getters & setters for class Model---------------------------------------------------------------------
    void setInput(Matrixd input)
    {
        _input = std::move(input);
        _layers[0]->setLayerActiveValues(_input);
    }
    void setLayerDerivation(size_t number_of_layer, Matrixd new_derivation_neurons_matrix)
    {
        _layers[number_of_layer]->setLayerDerivation(new_derivation_neurons_matrix);
    }
    void setLayerBias(size_t number_of_layer, Matrixd new_bias_matrix)
    {
        _layers[number_of_layer]->setLayerBias(new_bias_matrix);
    }
    void setLayerValues(size_t number_of_layer, Matrixd new_values_matrix)
    {
        _layers[number_of_layer]->setLayerValues(new_values_matrix);
    }
    void setLayerActiveValues(size_t number_of_layer, Matrixd new_active_values_matrix)
    {
        _layers[number_of_layer]->setLayerActiveValues(new_active_values_matrix);
    }
    void setLayerWeights(size_t number_of_layer, Matrixd new_weights_matrix)
    {
        _layers[number_of_layer]->setLayerWeights(new_weights_matrix);
    }
    void setLayerActivationFunction(size_t number_of_layer, ActivationFunction *new_activation_function)
    {
        _layers[number_of_layer]->setLayerActivationFunction(new_activation_function);
    }
    void setModelLossFunction(LossFunction *new_loss_function)
    {
        _loss_function = new_loss_function;
    }

    Matrixd getLayerBias(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerBias();
    }
    Matrixd getLayerValues(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerValues();
    }
    Matrixd getLayerActiveValues(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerActiveValues();
    }
    Matrixd getLayerWeights(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerWeights();
    }
    Matrixd getLayerBiasGradient(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerBiasGradient();
    }
    Matrixd getLayerWeightsGradient(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerWeightsGradient();
    }
    Matrixd getLayerDerivation(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerDerivation();
    }
    ActivationFunction *getLayerActivationFunction(size_t number_of_layer)
    {
        return _layers[number_of_layer]->getLayerActivationFunction();
    }
    LossFunction *getModelLossFunction()
    {
        return _loss_function;
    }
    Matrixd getOutput()
    {
        return _output;
    }
    Matrixd getInput()
    {
        return _input;
    }
    size_t getLayersSize()
    {
        return _layers.size();
    }

    // getters & setters for class Model---------------------------------------------------------------------
};

class Optimizer
{
protected:
    Model &_network;

public:
    explicit Optimizer(Model &network) : _network{network}
    {
        _network = network;
    }
    virtual void updateWeights(double learning_speed, double epoch) = 0;
};
class GD : public Optimizer
{
public:
    explicit GD(Model &network) : Optimizer(network) {}

    void updateWeights(double learning_speed = 0.05, double epoch = 1) override
    {
        for (int i = 1; i < _network.getLayersSize(); i++)
        {
            _network.setLayerWeights(i, _network.getLayerWeights(i) - _network.getLayerWeightsGradient(i) * learning_speed);
            _network.setLayerBias(i, _network.getLayerBias(i) - _network.getLayerBiasGradient(i) * learning_speed);
        }
    }
};
class ADAM : public Optimizer
{
public:
    Matrixd *_history_speed;
    Matrixd *_history_moment;
    double _gamma = 0.9;
    double _alfa = 0.999;
    double _epsilon = 1e-8;

    ADAM(Model &network) : Optimizer(network)
    {
        _network = network;
        _history_speed = new Matrixd[network.getLayersSize()];
        _history_moment = new Matrixd[network.getLayersSize()];

        for (int i = 1; i < network.getLayersSize(); ++i)
        {
            _history_speed[i] = network.getLayerWeights(i);
            _history_moment[i] = network.getLayerWeights(i);
            _history_speed[i].setZero();
            _history_moment[i].setZero();
        }
    }

    ~ADAM()
    {
        delete[] _history_speed;
        delete[] _history_moment;
    }

    void updateWeights(double learning_speed = 0.5, double epoch = 1)
    {

        for (int i = 0; i < _network.getLayersSize(); i++)
        {
            _history_speed[i] = (_gamma * _history_speed[i]) + ((1 - _gamma) * _network.getLayerWeightsGradient(i));
            _history_moment[i] = (_alfa * _history_moment[i]) + Eigen::MatrixXd((1 - _alfa) * _network.getLayerWeightsGradient(i).array().pow(2));
            _network.setLayerWeights(i, _network.getLayerWeights(i) - learning_speed * Eigen::MatrixXd((_history_speed[i] / (1 - pow(_gamma, epoch + 1))).array().cwiseQuotient((_history_moment[i] / (1 - pow(_alfa, epoch + 1))).array().sqrt() + _epsilon).array()));
        }
    }
};

int main()
{

    //srand(time(NULL));

    SquareErrorFunction square;
    LogisticFunction logistic;
    LinearFunction linear;
    SoftSignFunction ssf;
    SincFunction sinc;
    GhFunction gh;

    std::vector<std::shared_ptr<Layer>> layers{
        std::make_shared<SequentialLayer>(6,&logistic),
        std::make_shared<SequentialLayer>(3000, &logistic),
        std::make_shared<SequentialLayer>(2, &logistic),

    };

    Model network(&square, layers);

    Matrixd a(1, 6);
    a << 0.7, 0.7, 0.7, 0.7, 0.7, 0.7;
    Matrixd b(1, 2);
    b << 0.3, 0.3;



    GD gd(network);
    ADAM adam(network);

    int epoch = 1;

    while (true)
    {
        network.setInput(a);
        network.forwardPropogation();
        network.backPropogation(b);
        gd.updateWeights(0.2, epoch);

        std::cout << network.getOutput() << "\n";
        //system("pause");
        ++epoch;
    }

}
