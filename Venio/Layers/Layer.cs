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

        public abstract void ForwardPropogate(double[] last_layer_output);
        public abstract void BackPropogate(
            double[] nextDerivation,
            double[] nextValues,
            double[] nextWeights, 
            double[] last_active_values);

        public Layer(int inputSize, int outputSize , ActivationFunction activationFunction)
        {
            this.InputSize = inputSize;
            this.OutputSize = outputSize;
            this.activationFunction = activationFunction;

            this.bias = new double[inputSize];
            this.biasGrad = new double[inputSize];
            this.weights = new double[inputSize*outputSize];
            this.weightsGrad = new double[inputSize*outputSize];

            this.values = new double[inputSize];
            this.activeValues = new double[inputSize];
            this.valuesGrad = new double[inputSize];


            Utils.FillRandom(bias.AsSpan(), 0d, 1d);
            Utils.FillRandom(weights.AsSpan(), 0d, 1d);
            Utils.FillZero(biasGrad.AsSpan());
            Utils.FillZero(weightsGrad.AsSpan());
            Utils.FillZero(values.AsSpan());
            Utils.FillZero(activeValues.AsSpan());
            Utils.FillZero(valuesGrad.AsSpan());
        }

        public void Activate()
        {
            activationFunction.Activate(values, activeValues);
        }

    }
}
