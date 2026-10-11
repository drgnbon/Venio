using System;
using System.Collections.Generic;
using System.Text;

namespace Venio.Venio.LossFunctions
{
    public enum LossType
    {
        Square
    } 
    internal interface ILossFunction
    {
        public double GetAverageLoss(ReadOnlySpan<double> predicted, ReadOnlySpan<double> target);
        public double[] GetLossGrad(ReadOnlySpan<double> predicted, ReadOnlySpan<double> target);
    }
}
