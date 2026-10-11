using System;
using System.Collections.Generic;
using System.Text;

namespace Venio.Venio.ActivationFunctions
{
    internal sealed class Logistic : ActivationFunction
    {
        public override double Activate(double x) => 1d / (1d + Math.Exp(-x));

        public override double Derivate(double x)
        {
            double t = 1d / (1d + Math.Exp(-x));
            return t * (1d - t);
        }
    }
}
