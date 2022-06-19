using System;
using System.Collections.Generic;

namespace NeralNetwork
{
    public class Neyron
    {
        //Создание весов
        public List<double> Weights { get; }
        public List<double> Inputs { get; }
        public NeyronType NeyronType { get; }
        public double Output { get;private set; }
        public double Delta { get; private set; }

        public Neyron(int inputCount, NeyronType type = NeyronType.Normal)
        {
            NeyronType = type;
            Weights = new List<double>();
            Inputs = new List<double>();
            InitWeightsRandomValue(inputCount);
        }
        //сравниваем количество сигналов и количество весов

        //пишем основу
        private void InitWeightsRandomValue(int inputCount)
        {
            var rnd = new Random();
            for (int i = 0; i < inputCount; i++)
            {
                if (NeyronType == NeyronType.Input)
                {
                    Weights.Add(1);
                }
                else
                {
                    Weights.Add(rnd.NextDouble());
                }
                
                Inputs.Add(0);
            }
        }
        public double FeedForward(List<double> inputs)
        {
            for (int i = 0; i < inputs.Count; i++)
            {
                Inputs[i] = inputs[i];
            }

            var sum = 0.0;
            for (int i = 0; i < inputs.Count; i++)
            {
                sum += inputs[i] * Weights[i];
            }
            if (NeyronType != NeyronType.Input)
            {
                Output = Sigmoid(sum);
            }
            else
            {
                Output = sum;
            }
            return Output;
        }

        private double Sigmoid(double x)
        {
            var result = 1.0 / (1.0 + Math.Pow(Math.E, -x));
            return result;
        }

        private double SigmoidDx(double x)
        {
            var sigmoid = Sigmoid(x);
            var result = sigmoid * (1 - sigmoid);
            return result;
        }

        public void Learn(double error, double learningRate)
        {
            if (NeyronType == NeyronType.Input)
            {
                return;
            }

            Delta = error * SigmoidDx(Output);

            for (int i = 0; i < Weights.Count; i++)
            {
                var weight = Weights[i];
                var input = Inputs[i];

                var newWeight = weight - input * Delta * learningRate;
                Weights[i] = newWeight;
            }

           
        }

        public override string ToString()
        {
            return Output.ToString();
        }
    }
}
