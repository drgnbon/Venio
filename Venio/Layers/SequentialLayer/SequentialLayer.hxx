#include "../Layer/Layer.hxx"

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