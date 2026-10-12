using System;
using System.Collections.Generic;
using System.Text;

namespace Venio.Venio.Optimizers
{
    internal sealed class GD : IOptimizer
    {

        public void UpdateParameters(Span<double> weights, Span<double> bias,
                                     Span<double> weightsGrad, Span<double> biasGrad, double alpha)
        {
            if (weights.Length != weightsGrad.Length)
                throw new ArgumentException("Invalid weights size");
            if (bias.Length != biasGrad.Length)
                throw new ArgumentException("Invalid bias size");

            for (int i = 0; i < weights.Length; i++)
                weights[i] -= alpha * weightsGrad[i];

            for (int i = 0; i < bias.Length; i++)
                bias[i] -= alpha * biasGrad[i];

        }
    }
}
