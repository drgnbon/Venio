using System;
using System.Collections.Generic;
using System.Text;
using Venio.Venio.ActivationFunctions;
using Venio.Venio.Layers;
using Venio.Venio.LossFunctions;
using Venio.Venio.Optimizers;

namespace Venio.Venio.Model
{
    internal class Model
    {

        private ILossFunction lossFunction;
        private IOptimizer optimizer;
        private List<Layer> layers;
        public int Size { get; }

        private double[] _last_right_answer;


        private Model(
            List<Layer> layers, 
            ILossFunction lossFunction = null, 
            IOptimizer optimizer = null
        )
        {
            if (lossFunction == null) 
                lossFunction = new SquareLoss();
            if (optimizer == null)
                optimizer = new GD();

            this.lossFunction = lossFunction;
            this.layers = layers;
            this.optimizer = optimizer;
        }

        public void ForwardPropogate()
        {
            for (int i = 1; i < layers.Count; ++i)
            {
                layers[i].ForwardPropogate(layers[i - 1].ActiveValues);
            }
        }



        public class ModelBuilder
        {
            ILossFunction lossFunction;
            IOptimizer optimizer;
            List<Layer> layers;
            private ModelBuilder()
            {
                layers = new List<Layer>();
            }
            public static ModelBuilder Create()
            {
                return new ModelBuilder();
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

            public ModelBuilder Optimizer(IOptimizer optimizer)
            {
                this.optimizer = optimizer;
                return this;
            }

            public ModelBuilder LossFunction(LossType type)
            {
                lossFunction = type switch
                {
                    LossType.Square => new SquareLoss(),
                    _ => throw new ArgumentException("Unknown loss function")
                };
                return this;
            }
            public Model Build()
            {
                return new Model(layers, lossFunction, optimizer);
            }
        }
    }
}
