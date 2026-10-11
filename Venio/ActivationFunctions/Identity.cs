using System;
using System.Collections.Generic;
using System.Text;

namespace Venio.Venio.ActivationFunctions
{
    internal sealed class Identity : ActivationFunction
    {
        public override double Activate(double x) => x;

        public override double Derivate(double x) => 1d;

    }
}
