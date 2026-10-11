using System;
using System.Collections.Generic;
using System.Text;
using Venio.Venio.ActivationFunctions;

namespace Venio.Venio.Layers
{
    internal sealed class InputLayer : Layer
    {
        public InputLayer(int length)
            : base(0, length, new Identity()) { }
        public override void BackPropogate(double[] nextDerivation, double[] nextValues, double[] nextWeights, double[] last_active_values)
        {
            throw new NotImplementedException();
        }

        public override void ForwardPropogate(double[] input)
        {
            ArgumentNullException.ThrowIfNull(input);

            if (input.Length != OutputSize)
            {
                throw new ArgumentException(
                    $"Input layer expects {OutputSize} input values, but received {input.Length}.",
                    nameof(input));
            }

            Array.Copy(input, values, OutputSize);
            Activate();
        }
    }
}
