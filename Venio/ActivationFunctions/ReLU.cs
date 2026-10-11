using System;
using System.Collections.Generic;
using System.Text;

namespace Venio.Venio.ActivationFunctions
{
    internal sealed class ReLU : ActivationFunction
    {
        public override double Activate(double x) => Math.Max(0d,x);

        public override double Derivate(double x) => x > 0d ? 1d : 0d;
    }
}
