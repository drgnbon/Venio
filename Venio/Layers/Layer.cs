using System;
using System.Collections.Generic;
using System.Text;
using Venio.Venio.ActivationFunctions;

namespace Venio.Venio.Layers
{
    internal abstract class Layer
    {
        public int InputSize { get;}
        public int OutputSize { get;}
        protected double[] bias, biasGrad;
        protected double[] values, activeValues, valuesGrad;
        protected double[] weights, weightsGrad;

        public ActivationFunction activationFunction;

        public abstract void ForwardPropogate(double[] input);
        public abstract void BackPropogate(
            double[] nextDerivation,
            double[] nextValues,
            double[] nextWeights, 
            double[] last_active_values);

        public Layer(int inputSize, int outputSize , ActivationFunction activationFunction)
        {
            if (inputSize < 0)
                throw new ArgumentOutOfRangeException(nameof(inputSize), "Input size cannot be negative.");

            if (outputSize <= 0)
                throw new ArgumentOutOfRangeException(nameof(outputSize), "Output size must be greater than zero.");

            ArgumentNullException.ThrowIfNull(activationFunction);

            InputSize = inputSize;
            OutputSize = outputSize;
            this.activationFunction = activationFunction;

            bias = new double[OutputSize];
            biasGrad = new double[OutputSize];

            weights = new double[checked(InputSize * OutputSize)];
            weightsGrad = new double[weights.Length];

            values = new double[OutputSize];
            activeValues = new double[OutputSize];
            valuesGrad = new double[OutputSize];

            if (InputSize > 0)
            {
                Utils.FillRandom(bias.AsSpan(), 0d, 1d);
                Utils.FillRandom(weights.AsSpan(), 0d, 1d);
            }
        }

        public void Activate()
        {
            if (!activationFunction.Activate(values, activeValues))
            {
                throw new InvalidOperationException(
                    $"Activation function expected arrays of equal length ({OutputSize}).");
            }
        }

    }
}
