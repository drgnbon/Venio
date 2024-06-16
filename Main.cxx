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
class ActivationFunction{
public:
    ActivationFunction() = default;
    ~ActivationFunction() = default;

    virtual double getActiveValue(double x) = 0;
    virtual double getDerivateValue(double x) = 0;

    Matrixd getActiveMatrix(Matrixd matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = getActiveValue(matrix(i, j));
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

class LogisticFunction : public ActivationFunction{
public:
    double getActiveValue(double x) override
    {
        return 1.0 / (1.0 + exp(-x) );
    }

    double getDerivateValue(double x) override
    {
        return (1.0 / (1.0 + exp(-x) ))*(1 - (1.0 / (1.0 + exp(-x) )));
    }
};

class LinearFunction : public ActivationFunction{
public:
    double getActiveValue(double x) override
    {
        return x;
    }

    double getDerivateValue(double x) override
    {
        return 1;
    }
};

class SoftSignFunction : public ActivationFunction{
public:
    double getActiveValue(double x) override
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
class LossFunction{
public:
    virtual double getMediumLoss(Matrixd active_value,Matrixd right_answer) = 0;
    virtual Matrixd getDerivationLoss(Matrixd active_value,Matrixd right_answer) = 0;
};

class SquareErrorFunction : public LossFunction{
public:


    double getMediumLoss(Matrixd active_value,Matrixd right_answer) override
    {
        double SquareError = 0;
        for (int i = 0; i < active_value.rows(); ++i)
            for (int j = 0; j < active_value.cols(); ++j)
                SquareError += (active_value(i, j) - right_answer(i, j)) * (active_value(i, j) - right_answer(i, j));
        return SquareError;
    }
    //Probably warning---------------------------
    Matrixd getDerivationLoss(Matrixd active_value,Matrixd right_answer) override
    {
        return  2*(active_value-right_answer);
    }
};

//LossFunctions---------------------------------------------------------------------------------------------------------



//Layers----------------------------------------------------------------------------------------------------------------
class Layer{
public:
    Matrixd _values;
    Matrixd _active_values;
    Matrixd _derivation_neurons;
    Matrixd _weights;
    Matrixd _gradient;
    ActivationFunction *_activation_function;
    int _layer_size;


    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    Layer(int layer_size,ActivationFunction *activation_function)
    {
        _layer_size = layer_size;
        _activation_function = activation_function;
    }

    virtual ~Layer() {}

    void buildLayer(int input_size_of_last_layer)
    {
        _weights = Matrixd::Random(input_size_of_last_layer, _layer_size);
        _gradient = Matrixd::Zero(input_size_of_last_layer, _layer_size);
        _active_values = Matrixd::Zero(1, _layer_size);
        _values = Matrixd::Zero(1, _layer_size);
        _derivation_neurons = Matrixd::Zero(1, _layer_size);
    }

};



//work in progress--------------
class SequentialLayer : public Layer{
public:
    SequentialLayer(int layer_size, ActivationFunction *activation_function)
            : Layer(layer_size, activation_function){}
};

//work in progress--------------
class ConvolutionLayer : Layer{
public:
    ConvolutionLayer(int layer_size, ActivationFunction *activation_function)
            : Layer(layer_size, activation_function){}
};
//Layers----------------------------------------------------------------------------------------------------------------



class Model{

public:

    //to private
    std::vector<std::shared_ptr<Layer>> _layers;

    Model(std::vector<std::shared_ptr<Layer>> layers) {
        for(auto i:layers){
            push(i);
        }
    }

/*    Model(std::vector<std::shared_ptr<Layer>> layers)
    {
        _layers = std::move(layers);
        for(int i = 0;i < _layers.size();++i)
        {
            //_layers[i]->buildLayer(-layers[i+1]->_layer_size);
        }
    }*/

    void push(std::shared_ptr<Layer> layer){

        if(_layers.empty()){
            _layers.push_back(std::move(layer));
            //maybe trouble with layers

            _layers[0]->buildLayer(0);
            return;
        }

        _layers.push_back(std::move(layer));
        _layers[_layers.size()-1]->buildLayer(_layers[_layers.size()-2]->_layer_size);
    }


    ~Model(){
        _layers.clear();
    }


};



int main()
{
    LogisticFunction logistic;
    LinearFunction linear;
    std::vector<std::shared_ptr<Layer>> layers
    {
            std::make_shared<SequentialLayer>(6,&logistic),
            std::make_shared<SequentialLayer>(3,&logistic),
            std::make_shared<SequentialLayer>(2,&logistic),
    };
    Model network(layers);

    for(int i = 1;i < network._layers.size();++i){
        std::cout << network._layers[i]->_weights << "\n\n";
    }

}

