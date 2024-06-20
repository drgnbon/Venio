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
    void buildFirstLayer()
    {
        // Инициализация активных значений слоя нулями
        _active_values = Matrixd::Zero(1, _layer_size);

        // Инициализация значений слоя нулями
        _values = Matrixd::Zero(1, _layer_size);

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
        if (!(new_derivation_neurons_matrix.cols() == _derivation_neurons.cols() && new_derivation_neurons_matrix.rows() == _derivation_neurons.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerDerivation (Not equal size of matrix) - class Layer \n");
        }

        _derivation_neurons = new_derivation_neurons_matrix;
    }
    void setLayerBias(Matrixd new_bias_matrix)
    {
        if (!(new_bias_matrix.cols() == _bias.cols() && new_bias_matrix.rows() == _bias.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerBias (Not equal size of matrix) - class Layer \n");
        }

        _bias = new_bias_matrix;
    }
    void setLayerBiasGradient(Matrixd new_bias_gradient_matrix)
    {
        if (!(new_bias_gradient_matrix.cols() == _bias_gradient.cols() && new_bias_gradient_matrix.rows() == _bias_gradient.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerBiasGradient (Not equal size of matrix) - class Layer \n");
        }
        _bias_gradient = new_bias_gradient_matrix;
    }
    void setLayerValues(Matrixd new_values_matrix)
    {
        if (!(new_values_matrix.cols() == _values.cols() && new_values_matrix.rows() == _values.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerValues (Not equal size of matrix) - class Layer \n");
        }

        _values = new_values_matrix;
    }
    void setLayerActiveValues(Matrixd new_active_values_matrix)
    {
        if (!(new_active_values_matrix.cols() == _active_values.cols() && new_active_values_matrix.rows() == _active_values.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerActiveValues (Not equal size of matrix) - class Layer \n");
        }

        _active_values = new_active_values_matrix;
    }
    void setLayerWeights(Matrixd new_weights_matrix)
    {
        if (!(new_weights_matrix.cols() == _weights.cols() && new_weights_matrix.rows() == _weights.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerWeights (Not equal size of matrix) - class Layer \n");
        }

        _weights = new_weights_matrix;
    }
    void setLayerWeightsGradient(Matrixd new_weights_gradient_matrix)
    {

        if (!(new_weights_gradient_matrix.cols() == _weights_gradient.cols() && new_weights_gradient_matrix.rows() == _weights_gradient.rows()))
        {
            ErrorLogger::getInstance().logError("Error in setLayerWeightsGradient (Not equal size of matrix) - class Layer \n");
        }

        _weights_gradient = new_weights_gradient_matrix;
    }
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    void setLayerActivationFunction(ActivationFunction *new_activation_function)
    {
        // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
        try
        {
            _activation_function = new_activation_function;
        }
        catch (std::exception &e)
        {
            std::cerr << "Activation function error: " << e.what() << "\n";
        }
        // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    }

    // getters--------------------------------------------------------

    Matrixd getLayerDerivationMatrix(){
        return _activation_function->toDerivateMatrix(_values);
    }

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
    // Rename or function name or name of value
    Matrixd getLayerDerivation()
    {
        return _derivation_neurons;
    }
    // WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING - WARNING
    ActivationFunction *getLayerActivationFunction()
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
        _values = (last_layer_output * _weights) + _bias;
        activateLayer();
    }
    void backPropogateLayer(Matrixd next_layer_derivation, Matrixd next_layer_values, Matrixd next_layer_weights, Matrixd last_active_values) override
    {
        // Do work---------------------------------------------------------------------
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
