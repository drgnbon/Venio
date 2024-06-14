#include <Eigen\Core>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

using namespace Eigen;
typedef Eigen::MatrixXd Matrixd;

class ActivateFunction
{
    //сделать наследования и добавить много разных функций
    static double activate(double x)
    {
        return 1.0 / (1.0 + exp(-x) );
    }
    static double derivative(double x)
    {
        return (1.0 / (1.0 + exp(-x) ))*(1 - (1.0 / (1.0 + exp(-x) )));
    }
};






class Layer
{
public:
    int _layer_size;
    Matrixd _values;

    Layer(int layer_size): _layer_size{layer_size}{}

    virtual void buildLayer(int input_size_of_next_layer) = 0;


};

class InputLayer:public Layer
{
public:
    InputLayer(int layer_size) :Layer(layer_size){
        _layer_size = layer_size;

        std::cout << "Nulls in values created \n";
    }
//    explicit InputLayer()
//    {
//        std::cout << "ERR in inputlayer, empty builder \n";
//    }
    void buildLayer(int input_size_of_next_layer) override
    {
        _values = Matrixd::Zero(1,_layer_size);
    }
//
//    ~InputLayer();

};
class OutputLayer:public Layer
{
public:

    explicit OutputLayer(int layer_size) : Layer(layer_size)
    {
        _layer_size = layer_size;
        std::cout << "Nulls in values created \n";
    }



//    explicit OutputLayer()
//    {
//        std::cout << "ERR in outputLayer, empty builder \n ";
//    }
    void buildLayer(int input_size_of_next_layer) override
    {
        _values = Matrixd::Zero(1,_layer_size);
    }
//    ~OutputLayer();
};


class WorkLayer: public Layer {
public:
    Matrixd _active_values;
    Matrixd _derivation_neurons;
    Matrixd _weights;
    Matrixd _gradient;
    ActivateFunction _activation_function;


    WorkLayer(int layer_size, ActivateFunction activation_function)
            : Layer(layer_size), _activation_function(activation_function) {}







};


class SequentialLayer:public WorkLayer {
public:


    SequentialLayer(int layer_size, ActivateFunction activation_function) : WorkLayer(layer_size, activation_function)
    {
        _activation_function = activation_function;
        _layer_size = layer_size;
    }

    void buildLayer(int input_size_of_next_layer) override {
        _weights = Matrixd::Random(_layer_size, input_size_of_next_layer);
        _gradient = Matrixd::Zero(_layer_size, input_size_of_next_layer);
        _active_values = Matrixd::Zero(1, _layer_size);
        _values = Matrixd::Zero(1, _layer_size);
        _derivation_neurons = Matrixd::Zero(1, _layer_size);
    }


//    FullyConnectedLayer()
//    {
//        //_activation_function = &0;//set basic value,make in future///////////////////////////////////////////
////        std::cout << "fully connected layer constructor error \n";
////    }
//    ~FullyConnectedLayer();



//    void setLayers(std::initializer_list<int> structure)
//    {
//        std::vector<int> _structure = structure;
//        _active_values = Matrixd::Zero(1,_structure[0]);
//        _values = Matrixd::Zero(1,_structure[0]);
//        _derivation_neurons = Matrixd::Zero(1,_structure[0]);
//    }



    void getOutputLayer()
    {

    }
    void setInputLayer(Matrixd input_layer_values)
    {
        _values = std::move(input_layer_values);

    }

    void  calculateValues()
    {
        //we need calculate current layer right here

    }


//    Matrixd getActiveValues()
//    {
//        return  _active_values;
//    }




//    void activateLayer()
//    {
//        //_active_values = activation_function.activate(_values); //need make work in activation class structure
//    }
//
////    void calculateDerivation(Matrixd weights_this_layer,Matrixd derivation_next_layer,Matrixd values_next_layer,Matrixd _active_values_this_layer,std::shared_ptr<ActivationFunction> activation_function)
////    {
////        //_derivation_neurons = Matrixd( derivation_next_layer.array() * activation_function->getDerivateMatrix(values_next_layer).array()    ) * weights_this_layer.transpose();
////        //_gradient = (_active_values_this_layer.transpose() * Matrixd(derivation_next_layer.array() * activation_function->getDerivateMatrix(values_next_layer).array()));
////    }
//
//
//
//
//
//
//
//
//
//
//    ~FullyConnectedLayer();
};
//wip



//make in future
class ConvolutionLayer : Layer
{};
//make in future

class LossFunction
{
    double getMediumLoss(Matrixd active_value,Matrixd right_answer)
    {
        double SquareError = 0;
        for (int i = 0; i < active_value.rows(); ++i)
            for (int j = 0; j < active_value.cols(); ++j)
                SquareError += (active_value(i, j) - right_answer(i, j)) * (active_value(i, j) - right_answer(i, j));
        return SquareError;
    }


    Matrixd getDerivationLoss(Matrixd active_value,Matrixd right_answer)
    {
        return  2*(active_value-right_answer);
    }
};

class Optimizer
{

};

class Trainer
{

};;

class NeuralNetwork
{
    std::vector<Layer> _layers;//переделать наследование
public:
    NeuralNetwork(std::initializer_list<Layer> layers_list)
    {
        _layers = layers_list;

        for(int i = 0;i < layers_list.size();++i)
        {
            _layers[i].buildLayer(_layers[i+1]._layer_size);
        }
    }
};



int main()
{
    ActivateFunction sigmoid; //ахаха типо того
    NeuralNetwork network = {
            InputLayer(0, 5),
            SequentialLayer(5,sigmoid),
            SequentialLayer(3,sigmoid),
            SequentialLayer(2,sigmoid),
            SequentialLayer(1,sigmoid),
            OutputLayer(5)
            };














}

