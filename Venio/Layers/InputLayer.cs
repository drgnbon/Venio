using System;
using System.Collections.Generic;
using System.Text;
using Venio.Venio.ActivationFunctions;

namespace Venio.Venio.Layers
{
    internal sealed class InputLayer : Layer
    {
        public InputLayer(int length, ActivationFunction activationFunction)
            : base(0, length, activationFunction) { }
        public override void BackPropogate(double[] nextDerivation, double[] nextValues, double[] nextWeights, double[] last_active_values)
        {
            throw new NotImplementedException();
        }

        public override void ForwardPropogate(double[] last_layer_output)
        {
            throw new NotImplementedException();
        }
    }
}
