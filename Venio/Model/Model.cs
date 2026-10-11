using System;
using System.Collections.Generic;
using System.Text;
using Venio.Venio.ActivationFunctions;
using Venio.Venio.Layers;
using Venio.Venio.LossFunctions;

namespace Venio.Venio.Model
{
    internal class Model
    {

        private ILossFunction lossFunction;
        private List<Layer> layers;
        public int Size { get; }

        private double[] _last_right_answer;


        private Model(ILossFunction lossFunction,List<Layer> layers)
        {
            this.lossFunction = lossFunction;
            this.layers = layers;
        }

        public class ModelBuilder
        {
            ILossFunction lossFunction;
            List<Layer> layers;
            public ModelBuilder()
            {
                layers = new List<Layer>();
            }
            public ModelBuilder SequentialLayer(int length, ActivationFunction activationFunction)
            {
                if (layers.Count == 0)
                {
                    throw new ArgumentException("Empty layers exception, please insert input layer first");
                }
                int lastLayerSize = layers[^1].OutputSize;
                layers.Add(new SequentialLayer(lastLayerSize, length, activationFunction));
                return this;
            }

            public ModelBuilder InputLayer(int length)
            {
                layers.Add(new InputLayer(length));
                return this;
            }

            public ModelBuilder LossFunction(LossType type)
            {
                lossFunction = type switch
                {
                    LossType.Square =>  new SquareLoss(),
                    _ => throw new ArgumentException("Unknown loss function")
                };
                return this;
            }
            public Model Build()
            {
                return new Model(lossFunction, layers);
            }
    }
}
