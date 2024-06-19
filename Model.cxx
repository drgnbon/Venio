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

        if (_layers.size() == 0)
        {
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

        // Do work------------------------------------------------------------------
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