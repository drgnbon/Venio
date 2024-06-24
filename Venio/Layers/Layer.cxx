#include "Layer.hxx"

Layer::Layer(int layer_size, ActivationFunction *activation_function)
{
    _layer_size = layer_size;
    _activation_function = activation_function;
}

void Layer::buildLayer(int output_size_of_last_layer)
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

void Layer::buildFirstLayer()
{
    // Инициализация активных значений слоя нулями
    _active_values = Matrixd::Zero(1, _layer_size);

    // Инициализация значений слоя нулями
    _values = Matrixd::Zero(1, _layer_size);

    // Инициализация производных нейронов нулями
    _derivation_neurons = Matrixd::Zero(1, _layer_size);
}

void Layer::activateLayer()
{
    _active_values = _activation_function->toActivateMatrix(_values);
}

// getters & setters for class Layer---------------------------------------------------------------------

void Layer::setLayerDerivation(Matrixd new_derivation_neurons_matrix)
{
    if (!(new_derivation_neurons_matrix.cols() == _derivation_neurons.cols() && new_derivation_neurons_matrix.rows() == _derivation_neurons.rows()))
    {
        ErrorLogger::getInstance().logError("Error in setLayerDerivation (Not equal size of matrix) - class Layer \n");
    }

    _derivation_neurons = std::move(new_derivation_neurons_matrix);
}
void Layer::setLayerBias(Matrixd new_bias_matrix)
{
    if (!(new_bias_matrix.cols() == _bias.cols() && new_bias_matrix.rows() == _bias.rows()))
    {
        ErrorLogger::getInstance().logError("Error in setLayerBias (Not equal size of matrix) - class Layer \n");
    }

    _bias = std::move(new_bias_matrix);
}
void Layer::setLayerBiasGradient(Matrixd new_bias_gradient_matrix)
{
    if (!(new_bias_gradient_matrix.cols() == _bias_gradient.cols() && new_bias_gradient_matrix.rows() == _bias_gradient.rows()))
    {
        ErrorLogger::getInstance().logError("Error in setLayerBiasGradient (Not equal size of matrix) - class Layer \n");
    }
    _bias_gradient = std::move(new_bias_gradient_matrix);
}
void Layer::setLayerValues(Matrixd new_values_matrix)
{
    if (!(new_values_matrix.cols() == _values.cols() && new_values_matrix.rows() == _values.rows()))
    {
        ErrorLogger::getInstance().logError("Error in setLayerValues (Not equal size of matrix) - class Layer \n");
    }

    _values = std::move(new_values_matrix);
}
void Layer::setLayerActiveValues(Matrixd new_active_values_matrix)
{
    if (!(new_active_values_matrix.cols() == _active_values.cols() && new_active_values_matrix.rows() == _active_values.rows()))
    {
        ErrorLogger::getInstance().logError("Error in setLayerActiveValues (Not equal size of matrix) - class Layer \n");
    }

    _active_values = std::move(new_active_values_matrix);
}
void Layer::setLayerWeights(Matrixd new_weights_matrix)
{
    if (!(new_weights_matrix.cols() == _weights.cols() && new_weights_matrix.rows() == _weights.rows()))
    {
        ErrorLogger::getInstance().logError("Error in setLayerWeights (Not equal size of matrix) - class Layer \n");
    }

    _weights = std::move(new_weights_matrix);
}
void Layer::setLayerWeightsGradient(Matrixd new_weights_gradient_matrix)
{

    if (!(new_weights_gradient_matrix.cols() == _weights_gradient.cols() && new_weights_gradient_matrix.rows() == _weights_gradient.rows()))
    {
        ErrorLogger::getInstance().logError("Error in setLayerWeightsGradient (Not equal size of matrix) - class Layer \n");
    }

    _weights_gradient = std::move(new_weights_gradient_matrix);
}
// WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
void Layer::setLayerActivationFunction(ActivationFunction *new_activation_function)
{
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    try
    {
        _activation_function = new_activation_function;
    }
    catch (std::exception &e)
    {
        std::stringstream ss;
        ss << "Activation function error: " << e.what() << "\n";
        ErrorLogger::getInstance().logError(ss.str());
    }
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
}

// getters--------------------------------------------------------

Matrixd Layer::getLayerDerivationMatrix()
{
    return _activation_function->toDerivateMatrix(_values);
}

Matrixd Layer::getLayerBias()
{
    return _bias;
}
Matrixd Layer::getLayerValues()
{
    return _values;
}
Matrixd Layer::getLayerActiveValues()
{
    return _active_values;
}
Matrixd Layer::getLayerWeights()
{
    return _weights;
}
Matrixd Layer::getLayerBiasGradient()
{
    return _bias_gradient;
}
Matrixd Layer::getLayerWeightsGradient()
{
    return _weights_gradient;
}
// Rename or function name or name of value
Matrixd Layer::getLayerDerivation()
{
    return _derivation_neurons;
}
// WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
ActivationFunction* Layer::getLayerActivationFunction()
{
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    return _activation_function;
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
}
size_t Layer::getLayerSize() const
{
    return _layer_size;
}
// getters & setters for class Layer---------------------------------------------------------------------

// work in progress--------------
