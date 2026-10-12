using System;
using System.Collections.Generic;
using System.Text;

namespace Venio.Venio.Optimizers
{
    internal interface IOptimizer
    {

        public void UpdateParameters(Span<double> weights, Span<double> bias,
                                     Span<double> weightsGrad, Span<double> biasGrad, double alpha);
    }
}
