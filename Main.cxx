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
class ActivationFunction {
public:
    ActivationFunction() = default;

    ~ActivationFunction() = default;

    virtual double toActivateValue(double x) = 0;

    virtual double toDerivateValue(double x) = 0;

    Matrixd toActivateMatrix(Matrixd matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = toActivateValue(matrix(i, j));
            }
        }
        return matrix;
    }

    Matrixd toDerivateMatrix(Matrixd matrix) {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                matrix(i, j) = toDerivateValue(matrix(i, j));
            }
        }
        return matrix;
    }
};

class LogisticFunction : public ActivationFunction {
public:
    double toActivateValue(double x) override {
        return 1.0 / (1.0 + exp(-x));
    }

    double toDerivateValue(double x) override {
        return (1.0 / (1.0 + exp(-x))) * (1 - (1.0 / (1.0 + exp(-x))));
    }
};

class LinearFunction : public ActivationFunction {
public:
    double toActivateValue(double x) override {
        return x;
    }

    double toDerivateValue(double x) override {
        return 1;
    }
};

class SoftSignFunction : public ActivationFunction {
public:
    double toActivateValue(double x) override {
        return x / (1 + abs(x));
    }

    double toDerivateValue(double x) override {
        return 1 / pow(1 + abs(x), 2);
    }
};
//ActivationFunctions---------------------------------------------------------------------------------------------------

//LossFunctions---------------------------------------------------------------------------------------------------------
class LossFunction {
public:
    virtual double getMediumLoss(Matrixd active_value, Matrixd right_answer) = 0;

    virtual Matrixd getDerivationLoss(Matrixd active_value, Matrixd right_answer) = 0;
};

class SquareErrorFunction : public LossFunction {
public:


    double getMediumLoss(Matrixd active_value, Matrixd right_answer) override {
        double SquareError = 0;
        for (int i = 0; i < active_value.rows(); ++i)
            for (int j = 0; j < active_value.cols(); ++j)
                SquareError += (active_value(i, j) - right_answer(i, j)) * (active_value(i, j) - right_answer(i, j));
        return SquareError;
    }

    //Probably warning---------------------------
    Matrixd getDerivationLoss(Matrixd active_value, Matrixd right_answer) override {
        return 2 * (active_value - right_answer);
    }
};

//LossFunctions---------------------------------------------------------------------------------------------------------



//Layers----------------------------------------------------------------------------------------------------------------
class Layer {
public:
    Matrixd _bias,_bias_gradient;
    Matrixd _values,_active_values;
    Matrixd _derivation_neurons;
    Matrixd _weights,_weights_gradient;
    ActivationFunction *_activation_function;
    int _layer_size;


    Layer(int layer_size, ActivationFunction *activation_function) {
        _layer_size = layer_size;
        _activation_function = activation_function;
    }

    virtual ~Layer() {}


    //warning rebasing to next layer
    void buildLayer(int input_size_of_next_layer) {
        _weights = Matrixd::Random(_layer_size, input_size_of_next_layer);
        _weights_gradient = Matrixd::Zero(_layer_size, input_size_of_next_layer);
        _active_values = Matrixd::Zero(1, _layer_size);
        _values = Matrixd::Zero(1, _layer_size);
        _bias = Matrixd::Random(1, _layer_size);
        _derivation_neurons = Matrixd::Zero(1, _layer_size);
    }

    void setLayerValues(Matrixd values) {

        //Exeption if's {
        if( !(values.cols() == _values.cols()
              && values.rows() == _values.rows()) ){
            std::cout <<"Error in setting layer (Not equal size of matrix) \n";
            system("pause");
            exit(0);
        }
        // }

        _values = values;
    }
    void setLayerDerivation(Matrixd derivation_neurons) {

        //Exeption if's {
        if( !(derivation_neurons.cols() == _derivation_neurons.cols()
              && derivation_neurons.rows() == _derivation_neurons.rows()) ){
            std::cout <<"Error in setting layer (Not equal size of matrix) \n";
            system("pause");
            exit(0);
        }
        // }

        _derivation_neurons = derivation_neurons;
    }
    Matrixd getLayerActiveValues(){
        return _active_values;
    }
    Matrixd getLayerDerivation(){
        return _derivation_neurons;
    }
    Matrixd getLayerValues(){
        return _values;
    }
    Matrixd getWeights(){
        return _weights;
    }
    void activateLayer(){
        _active_values = _activation_function->toActivateMatrix(_values);
    }

    virtual void propogateLayer(Matrixd last_layer_output,Matrixd last_layer_weights) = 0;
    virtual void backPropogateLayer(Matrixd derivation_neurons_next_layer, Matrixd next_layer_values) = 0;


};


//work in progress--------------
class SequentialLayer : public Layer {
public:
    SequentialLayer(int layer_size, ActivationFunction *activation_function)
            : Layer(layer_size, activation_function) {}

    void propogateLayer(Matrixd last_layer_output,Matrixd last_layer_weights) override{
        _values = last_layer_output*last_layer_weights + _bias;
        activateLayer();
    }

    //work----------------------------------------------------------------------------------------------------
    void backPropogateLayer(Matrixd derivation_neurons_next_layer, Matrixd next_layer_values) override{

        Matrixd next_layer_derivation = Matrixd( derivation_neurons_next_layer.array() * _activation_function->toDerivateMatrix(next_layer_values).array()    );

        _derivation_neurons = next_layer_derivation * _weights.transpose();
        _weights_gradient = _active_values.transpose() * next_layer_derivation;
    }


};

//work in progress--------------
class ConvolutionLayer : Layer {
public:
    ConvolutionLayer(int layer_size, ActivationFunction *activation_function)
            : Layer(layer_size, activation_function) {}

    void propogateLayer(Matrixd last_layer_output,Matrixd last_layer_weights) override{
        //Work in progress--------------------------
    }
    void backPropogateLayer(Matrixd derivation_neurons_next_layer, Matrixd next_layer_values) override{
        //Work in progress--------------------------
    }
};
//Layers----------------------------------------------------------------------------------------------------------------



class Model {

    bool _is_compiled;
    LossFunction *_loss_function;
    Matrixd _input, _output;
    std::vector<std::shared_ptr<Layer>> _layers;

public:



    Model(LossFunction *loss_function,std::vector<std::shared_ptr<Layer>> layers) {

        for (auto i: layers) {
            push(i);
        }

        compile();
        _input = Matrixd::Zero(1, _layers[0]->_layer_size);
        _output = Matrixd::Zero(1, _layers[_layers.size() - 1]->_layer_size);
        _loss_function = loss_function;
    }
    ~Model() {
        _layers.clear();
    }

/*    Model(std::vector<std::shared_ptr<Layer>> layers)
    {
        _layers = std::move(layers);
        for(int i = 0;i < _layers.size();++i)
        {
            //_layers[i]->buildLayer(-layers[i+1]->_layer_size);
        }
    }*/


    void Log() {

        std::cout << "Input size : " << _input.cols() << ", Input: " << _input << "\n";

        printf("Number of layers: %d \n", _layers.size());
        std::cout << "Layers: \n";

        int k = 1;
        for (auto i: _layers) {
            printf("\tLayer num: %d,Layer size: %d,Layer active values: ", k++, i->_layer_size);
            std::cout << i->_active_values << "\n";
            std::cout << "\tLayer derivations: " << i->_derivation_neurons << "\n\n";

        }

        std::cout << "Output size : " << _output.cols() << ", Output: " << _output << "\n";
    }
    void compile(){
        if(_is_compiled) return;
        _is_compiled = true;
        _layers[_layers.size() - 1]->buildLayer(0);
    }

    void setInput(Matrixd input) {

        //Exeption if's {
        /*if( !(_input.cols() == input.cols()
              && _input.rows() == input.rows()) ){
            std::cout << "You are trying to assign an input that is not equal in size to the input layer \n";
            system("pause");
            exit(0);
        }*/
        // }


        //add if's
        _input = std::move(input);
        _layers[0]->_active_values = _input;
    }
    Matrixd getOutput() {
        return _output;
    }
    void push(std::shared_ptr<Layer> layer) {

        _is_compiled = false;

        if (_layers.empty()) {
            _layers.push_back(std::move(layer));
            return;
        }

        _layers.push_back(std::move(layer));
        _layers[_layers.size() - 2]->buildLayer(_layers[_layers.size() - 1]->_layer_size);
    }


    //work--------------------------------------------------------------------
    void forwardPropogation(){

        if(!_is_compiled){
            compile();
        }

        for(int i = 1;i < _layers.size();++i){
            _layers[i]->propogateLayer(_layers[i-1]->getLayerActiveValues(),_layers[i-1]->getWeights());

        }

        //Add activation to output (maybe softmax)-----------------
        _output = _layers[_layers.size()-1]->getLayerActiveValues();
    }
    void backPropogation(Matrixd right_answer){

        if(!_is_compiled){
            compile();
        }

        _layers[_layers.size()-1]->_derivation_neurons = _loss_function->getDerivationLoss(_output, right_answer);

        for(long long i = _layers.size()-2; i >= 0; --i)
        {
            //WARNING update to new many activation functions on different layers
            _layers[i]->backPropogateLayer( _layers[i+1]->_derivation_neurons, _layers[i+1]->_values);
        }
    }

};


int main() {
    //srand(time(NULL));

    SquareErrorFunction square;
    LogisticFunction logistic;
    LinearFunction linear;
    std::vector<std::shared_ptr<Layer>> layers
            {
                    std::make_shared<SequentialLayer>(6, &logistic),
                    std::make_shared<SequentialLayer>(3, &logistic),
                    std::make_shared<SequentialLayer>(2, &logistic),
            };
    Model network(&square,layers);


    Matrixd a(1, 6);
    Matrixd b(1,2);

    for (int i = 0; i < 6; ++i) {
        std::cin >> a(i);
    }
    for (int i = 0; i < 2; ++i) {
        std::cin >> b(i);
    }

    network.setInput(a);
    network.forwardPropogation();
    network.backPropogation(b);
    network.Log();


    system("pause");

}

