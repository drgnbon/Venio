using System;
using System.Buffers;
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



        public class MathBuilder : IDisposable
        {
            private double[] _buffer;
            private int _length;

            private MathBuilder(int length)
            {
                _length = length;
                // Берем массив из пула (или создаем новый, если пул пуст)
                _buffer = ArrayPool<double>.Shared.Rent(length);
            }

            public static MathBuilder Create(int length)
            {
                return new MathBuilder(length);
            }

            public static MathBuilder From(ReadOnlySpan<double> input)
            {
                var builder = new MathBuilder(input.Length);
                input.CopyTo(builder._buffer);
                return builder;
            }

            public MathBuilder Mult(ReadOnlySpan<double> multiplier)
            {
                for (int i = 0; i < _length; i++)
                    _buffer[i] *= multiplier[i];
                return this;
            }

            public MathBuilder Add(ReadOnlySpan<double> other)
            {
                for (int i = 0; i < _length; i++)
                    _buffer[i] += other[i];
                return this;
            }

            public MathBuilder Subtract(ReadOnlySpan<double> other)
            {
                for (int i = 0; i < _length; i++)
                    _buffer[i] -= other[i];
                return this;
            }

            public MathBuilder Scale(double scalar)
            {
                for (int i = 0; i < _length; i++)
                    _buffer[i] *= scalar;

                return this;
            }
            public double Sum()
            {
                double sum = 0;
                for (int i = 0; i < _length; i++)
                    sum += _buffer[i];
                return sum;
            }
            public MathBuilder Square()
            {
                for (int i = 0; i < _length; i++)
                    _buffer[i] *= _buffer[i];

                return this;
            }

            public double Mean()
            {
                return Sum() / _length;
            }


            public double[] Result
            {
                get
                {
                    var result = new double[_length];
                    Array.Copy(_buffer, result, _length);
                    return result;
                }
            }

            // ВАЖНО: Нужно вернуть массив в пул!
            public void Dispose()
                {
                    if (_buffer != null)
                    {
                        ArrayPool<double>.Shared.Return(_buffer);
                        _buffer = null;
                    }
                }
            }


        public static void MultMatrix(
           ReadOnlySpan<double> input,
           ReadOnlySpan<double> weights,
           Span<double> result,
           int inputSize,
           int outputSize)
        {
            if (input.Length != inputSize)
                throw new ArgumentException($"Input length mismatch");
            if (weights.Length != inputSize * outputSize)
                throw new ArgumentException($"Weights length mismatch");
            if (result.Length != outputSize)
                throw new ArgumentException($"Result length mismatch");

            for (int i = 0; i < outputSize; i++)
            {
                double sum = 0;
                for (int j = 0; j < inputSize; j++)
                {
                    sum += input[j] * weights[j * outputSize + i];
                }
                result[i] = sum;
            }
        }

        // Добавление bias: result += bias
        public static void AddBias(Span<double> result, ReadOnlySpan<double> bias)
        {
            if (result.Length != bias.Length)
                throw new ArgumentException("Length mismatch");

            for (int i = 0; i < result.Length; i++)
                result[i] += bias[i];
        }


    }
}
