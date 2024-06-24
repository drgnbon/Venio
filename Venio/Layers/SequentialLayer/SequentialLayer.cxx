#include "SequentialLayer.hxx"

void SequentialLayer::propogateLayer(Matrixd last_layer_output)
{
    _values = (last_layer_output * _weights) + _bias;
    activateLayer();
}
void SequentialLayer::backPropogateLayer(Matrixd next_layer_derivation, Matrixd next_layer_values, Matrixd next_layer_weights, Matrixd last_active_values)
{
    // Do work---------------------------------------------------------------------
}
