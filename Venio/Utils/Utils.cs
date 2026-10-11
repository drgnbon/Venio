using System;
using System.Collections.Generic;
using System.Data.SqlTypes;
using System.Text;
using System.Xml;
using static Venio.Venio.ActivationFunctions.ActivationFunction;

namespace Venio.Venio
{
    internal class Utils
    {
        public static bool TryApplyFunction(ReadOnlySpan<double> input, Span<double> output, Func<double, double> func)
        {

            if (input.Length != output.Length)
                return false;

            for (int i = 0; i < input.Length; i++)
            {
                output[i] = func(input[i]);
            }

            return true;
        }

        internal static void FillRandom(Span<double> destination,double from, double to)
        {
            double range = to - from;
            var rng = Random.Shared;

            for (int i = 0; i < destination.Length; i++)
                destination[i] = from + rng.NextDouble() * range;
        }

        internal static void FillZero(Span<double> destination)
        {    
            for (int i = 0; i < destination.Length; i++)
                destination[i] = 0d;
        }
    }
}
