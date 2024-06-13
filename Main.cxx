#include <Eigen\Core>
#include <iostream>
#include <initializer_list>
#include <memory>
#include <vector>

using namespace Eigen;
typedef Eigen::MatrixXd Matrixd;

class ActivateFunction
{
};

//wip
class Layer
{
    Layer();
    ~Layer();
};

class FullyConnectedLayer
{
    Matrixd _active_values;
    Matrixd _values;
    Matrixd _derivation_neurons;
    //ActivateFunction& _activation_function;/////////////////////////////////////////////////////////

    FullyConnectedLayer(std::initializer_list<int> structure)
    {
        //_activation_function = &0;//set basic value,make in future//////////////////////////////////////////
        std::vector<int> _structure = structure;
        _active_values = Matrixd::Zero(1,_structure[0]);
        _values = Matrixd::Zero(1,_structure[0]);
        _derivation_neurons = Matrixd::Zero(1,_structure[0]);
    }
    FullyConnectedLayer()
    {
        //_activation_function = &0;//set basic value,make in future///////////////////////////////////////////
        std::cout << "fully connected layer constructor error \n";
    }

    void setLayers(std::initializer_list<int> structure)
    {
        std::vector<int> _structure = structure;
        _active_values = Matrixd::Zero(1,_structure[0]);
        _values = Matrixd::Zero(1,_structure[0]);
        _derivation_neurons = Matrixd::Zero(1,_structure[0]);
    }

    void  calculateValues()
    {
        //we need calculate current layer right here

    }
    Matrixd getActiveValues()
    {
        return  _active_values;
    }


    void activateLayer()
    {
        _active_values = activation_function.activate(_values); //need make work in activation class structure
    }

//    void calculateDerivation(Matrixd weights_this_layer,Matrixd derivation_next_layer,Matrixd values_next_layer,Matrixd _active_values_this_layer,std::shared_ptr<ActivationFunction> activation_function)
//    {
//        //_derivation_neurons = Matrixd( derivation_next_layer.array() * activation_function->getDerivateMatrix(values_next_layer).array()    ) * weights_this_layer.transpose();
//        //_gradient = (_active_values_this_layer.transpose() * Matrixd(derivation_next_layer.array() * activation_function->getDerivateMatrix(values_next_layer).array()));
//    }










    ~FullyConnectedLayer();

public:

private:

};
//wip



//make in future
class ConvolutionLayer : Layer
{};
//make in future





class NeuralNetwork
{
};



class ActivateFunction
{
};

class LossFunction
{
};

class Optimizer
{

};

class Trainer
{

};;





int main()
{

}

