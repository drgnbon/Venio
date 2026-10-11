using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Text;

namespace Venio.Venio.ActivationFunctions
{
    internal abstract class ActivationFunction
    {

        public abstract double Activate(double x);
        public abstract double Derivate(double x);


        public bool Activate(ReadOnlySpan<double> input, Span<double> output) 
            => Utils.TryApplyFunction(input,output,Activate);

        public bool Derivate(ReadOnlySpan<double> input, Span<double> output)
            => Utils.TryApplyFunction(input, output, Derivate);

    }
}
