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



//ActivationFunctions---------------------------------------------------------------------------------------------------
class ActivationFunction
        {
        public:
    ActivationFunction() = default;
    ~ActivationFunction() = default;

    virtual double getActivateValue(double x) = 0;
    virtual double getDerivateValue(double x) = 0;

    Matrixd getActivateMatrix(Matrixd matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = getActivateValue(matrix(i, j));
            }
        }
        return matrix;
    }

    Matrixd getDerivateMatrix(Matrixd matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = getDerivateValue(matrix(i, j));
            }
        }
        return matrix;
    }
        };

class LogisticFunction : public ActivationFunction
        {
        public:
    double getActivateValue(double x) override
    {
        return 1.0 / (1.0 + exp(-x) );
    }

    double getDerivateValue(double x) override
    {
        return (1.0 / (1.0 + exp(-x) ))*(1 - (1.0 / (1.0 + exp(-x) )));
    }
        };

class SoftSignFunction : public ActivationFunction
{
public:
    double getActivateValue(double x) override
    {
        return x/(1+abs(x));
    }

    double getDerivateValue(double x) override
    {
        return 1/pow(1+abs(x),2);
    }
};
//ActivationFunctions---------------------------------------------------------------------------------------------------

//LossFunctions---------------------------------------------------------------------------------------------------------
class LossFunction
        {
        public:
    virtual double getMediumLoss(Matrixd active_value,Matrixd right_answer) = 0;
    virtual Matrixd getDerivationLoss(Matrixd active_value,Matrixd right_answer) = 0;
        };

class SquareErrorFunction : public LossFunction
        {
        public:
    double getMediumLoss(Matrixd active_value,Matrixd right_answer) override
    {
        double SquareError = 0;
        for (int i = 0; i < active_value.rows(); ++i)
            for (int j = 0; j < active_value.cols(); ++j)
                SquareError += (active_value(i, j) - right_answer(i, j)) * (active_value(i, j) - right_answer(i, j));
        return SquareError;
    }
    Matrixd getDerivationLoss(Matrixd active_value,Matrixd right_answer) override
    {
        return  2*(active_value-right_answer);
    }
        };

//LossFunctions---------------------------------------------------------------------------------------------------------



//Layers----------------------------------------------------------------------------------------------------------------
class Layer
        {
        public:
    int _layer_size;
    Matrixd _values;
    ActivationFunction *_activation_function;

    Layer(int layer_size)
    {
        _layer_size = layer_size;
    }
    Layer(int layer_size,ActivationFunction *activation_function)
    {
        _layer_size = layer_size;
    }
    virtual ~Layer() {}

    virtual void buildLayer(int input_size_of_next_layer) = 0;

        };

class WorkingLayer : public Layer
        {
        public:
    Matrixd _active_values;
    Matrixd _derivation_neurons;
    Matrixd _weights;
    Matrixd _gradient;

    WorkingLayer(int layer_size,ActivationFunction *activation_function) : Layer(layer_size,activation_function)
    {
        _layer_size = layer_size;
        _activation_function = activation_function;
    }

    void buildLayer(int input_size_of_next_layer) override
    {
        _weights = Matrixd::Random(_layer_size, input_size_of_next_layer);
        _gradient = Matrixd::Zero(_layer_size, input_size_of_next_layer);
        _active_values = Matrixd::Zero(1, _layer_size);
        _values = Matrixd::Zero(1, _layer_size);
        _derivation_neurons = Matrixd::Zero(1, _layer_size);
    }
        };

class InputLayer : public Layer
        {
        public:
    explicit InputLayer(int layer_size) :Layer(layer_size){}

    void buildLayer(int input_size_of_next_layer) override {
        _values = Matrixd::Zero(1, _layer_size);
    }
        };

class OutputLayer : public Layer
        {
        public:

    explicit OutputLayer(int layer_size) : Layer(layer_size){}

    void buildLayer(int input_size_of_next_layer) override {
        _values = Matrixd::Zero(1,_layer_size);
    }
        };

class SequentialLayer : public WorkingLayer
        {
        public:
    SequentialLayer(int layer_size, ActivationFunction *activation_function)
            : WorkingLayer(layer_size, activation_function){}
        };

class ConvolutionLayer : WorkingLayer
        {
        public:
    ConvolutionLayer(int layer_size, ActivationFunction *activation_function)
            : WorkingLayer(layer_size, activation_function){}
        };
//Layers----------------------------------------------------------------------------------------------------------------





class Model
{
    std::vector<std::shared_ptr<Layer>> _layers;
public:
    Model(std::vector<std::shared_ptr<Layer>> layers)
    {
        _layers = layers;
        for(int i = 0;i < _layers.size();++i)
        {
            //_layers[i]->buildLayer(-layers[i+1]->_layer_size);
        }
    }
};



int main()
{
    LogisticFunction logistic;
    std::vector<std::shared_ptr<Layer>> layers
    {
            std::make_shared<InputLayer>(5),
            std::make_shared<SequentialLayer>(6,&logistic),
            std::make_shared<OutputLayer>(5)
    };
    Model network(layers);

}

