#pragma once

#include "LossFunction.hxx"
#include "ActivationFunction.hxx"
#include "Layer.hxx"
#include "Config.hxx"
#include <memory>

class Model
{
private:
    LossFunction *_loss_function;
    std::vector<std::shared_ptr<Layer>> _layers;
    size_t _model_size;
    Matrixd _last_right_answer;

public:
    Model(LossFunction *loss_function, const std::vector<std::shared_ptr<Layer>> &layers);
    ~Model();

    void addLayer(std::shared_ptr<Layer> layer);

    void Log();

    void forwardPropogation();
    void backPropogation(Matrixd right_answer);
    void backPropogation();

    // getters & setters for class Model---------------------------------------------------------------------
    void setInput(Matrixd input);
    void setLayerDerivation(size_t number_of_layer, Matrixd new_derivation_neurons_matrix);
    void setLayerBias(size_t number_of_layer, Matrixd new_bias_matrix);
    void setLayerValues(size_t number_of_layer, Matrixd new_values_matrix);
    void setLayerActiveValues(size_t number_of_layer, Matrixd new_active_values_matrix);
    void setLayerWeights(size_t number_of_layer, Matrixd new_weights_matrix);
    void setLayerActivationFunction(size_t number_of_layer, ActivationFunction *new_activation_function);
    void setModelLossFunction(LossFunction *new_loss_function);

    Matrixd getDerivationLossForLastLayer(Matrixd right_answer);

    Matrixd getLayerBias(size_t number_of_layer);
    Matrixd getLayerValues(size_t number_of_layer);
    Matrixd getLayerActiveValues(size_t number_of_layer);
    Matrixd getLayerWeights(size_t number_of_layer);
    Matrixd getLayerBiasGradient(size_t number_of_layer);
    Matrixd getLayerWeightsGradient(size_t number_of_layer);
    Matrixd getLayerDerivation(size_t number_of_layer);
    ActivationFunction *getLayerActivationFunction(size_t number_of_layer);
    LossFunction *getModelLossFunction();
    Matrixd getOutput();
    Matrixd getInput();
    size_t getLayersSize();

    // getters & setters for class Model---------------------------------------------------------------------
};