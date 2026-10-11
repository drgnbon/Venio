using System;
using System.Collections.Generic;
using System.Text;
using Venio.Venio.ActivationFunctions;

namespace Venio.Venio.Layers
{
    internal sealed class SequentialLayer : Layer
    {
        public SequentialLayer(int inputSize, int outputSize, ActivationFunction activationFunction)
            : base(inputSize, outputSize, activationFunction) { }

        public override void BackPropogate(double[] nextDerivation, double[] nextValues, double[] nextWeights, double[] last_active_values)
        {
            throw new NotImplementedException();
        }

        public override void ForwardPropogate(double[] lastOutput)
        {

            //add without allocation method
            values = Utils.MathBuilder
                .From(lastOutput)
                .Mult(weights)
                .Add(bias)
                .Result;

            Activate();
        }
    }
}
