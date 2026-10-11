using System;
using System.Collections.Generic;
using System.Text;

namespace Venio.Venio.ActivationFunctions
{
    internal sealed class Tanh : ActivationFunction
    {
        public override double Activate(double x) => Math.Tanh(x);

        public override double Derivate(double x)
        {
            double t = Math.Tanh(x);
            return 1 - t*t;
        }
    }
}
