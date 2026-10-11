using System;
using System.Collections.Generic;
using System.Text;
using static Venio.Venio.Utils;

namespace Venio.Venio.LossFunctions
{
    internal sealed class SquareLoss : ILossFunction
    {
        public double GetAverageLoss(ReadOnlySpan<double> predicted, ReadOnlySpan<double> target)
        {
            return MathBuilder.From(predicted)
                .Subtract(target)      
                .Square()              
                .Mean();              
        }

        public double[] GetLossGrad(ReadOnlySpan<double> predicted, ReadOnlySpan<double> target)
        {

            return MathBuilder.From(predicted)
                .Subtract(target)    
                .Scale(2d / predicted.Length)         
                .Result;              
        }

    }
}
