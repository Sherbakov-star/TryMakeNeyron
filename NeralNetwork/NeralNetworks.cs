using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeralNetwork
{
    public class NeralNetworks
    {
        public Topology Topology { get; }
        public List<Layer> Layers { get; }

        public NeralNetworks(Topology topology)
        {
            Topology = topology;

            Layers = new List<Layer>();

            CreateInputLayer();
            CreateHiddenLayers();
            CreateOutputLayer();
        }

        public Neyron FeedForward(params double[] inputSignals)
        {
         
            //Движение по нейронке
            SendsSignalsToInputNeyrons(inputSignals);
            FeedForwardAllLayersAfterInput();

            if (Topology.OutputCount == 1)
            {
                return Layers.Last().Neyrons[0];
            }
            else
            {
                return Layers.Last().Neyrons.OrderByDescending(n => n.Output).First();
            }
        }

        public double Learn(double[] expected, double[,] inputs,int epoch)
        {
            var signals = Normalization(inputs);

            var error = 0.0;
            for (int i = 0; i < epoch; i++)
            {
                for (int j = 0; j < expected.Length; j++)
                {
                    var output = expected[j];
                    var input = GetRow(inputs, j);

                    error += Backpropagation(output, input);
                }
            }
            var result = error/ epoch;
            return result;
        }

        public static double[] GetRow(double[,] matrix,int row)
        {
            var columns = matrix.GetLength(1);
            var array = new double[columns];
            for (int i = 0; i < columns; ++i)
                array[i] = matrix[row,i];
            return array;
        }
        
        private double[,] Scalling(double[,] inputs) // алгоритм масштабирования
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                var min = inputs[0, column];
                var max = inputs[0, column];

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    var item = inputs[row, column];

                    if (item < min)
                    {
                        min = item;
                    }

                    if (item > max)
                    {
                        max = item;
                    }
                }

                var divider = max - min;

                for (int row = 1; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - min) / divider;
                }
            }

            return result;
        }

        private double[,] Normalization(double[,] inputs) // алгоритм реализации
        {
            var result = new double[inputs.GetLength(0), inputs.GetLength(1)];

            for (int column = 0; column < inputs.GetLength(1); column++)
            {
                // Среднее значение сигнала нейрона
                var sum = 0.0;

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    sum += inputs[row, column];
                }
                var average = sum / inputs.GetLength(0);
                // Стандартное квадратичное отклонение
                var error = 0.0;
                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    error += Math.Pow((inputs[row, column] - average), 2);
                }
                var standartError = Math.Sqrt(error / inputs.GetLength(0));

                for (int row = 0; row < inputs.GetLength(0); row++)
                {
                    result[row, column] = (inputs[row, column] - average) / standartError;
                }
            }

            return result;
        }

            private double Backpropagation(double exprected,params double[] inputs)
            {
                var actual = FeedForward(inputs).Output;

                var difference = actual - exprected;

                foreach (var neyron in Layers.Last().Neyrons)
                {
                neyron.Learn(difference, Topology.LearningRate);
                }
    
                for (int j = Layers.Count -2; j >= 0; j--)
                {
                    var layer = Layers[j];
                    var previousLayer = Layers[j + 1];

                    for (int i = 0; i < layer.NeyronCount; i++)
                    {
                        var neyron = layer.Neyrons[i];

                        for (int k = 0; k < previousLayer.NeyronCount; k++)
                        {
                            var previousNeyron = previousLayer.Neyrons[k];
                            var error = previousNeyron.Weights[i] * previousNeyron.Delta;
                            neyron.Learn(error, Topology.LearningRate);
                        }
                    }
                }
                var result = difference * difference;
                return result;
            }

        private void FeedForwardAllLayersAfterInput()
        {
            for (int i = 1; i < Layers.Count; i++)
            {
                var layer = Layers[i];
                var previousLayerSignals = Layers[i - 1].GetSignals();

                foreach (var neyron in layer.Neyrons)
                {
                    neyron.FeedForward(previousLayerSignals);
                }
            }
        }

        private void SendsSignalsToInputNeyrons(params double[] inputSignals)
        {
            //Движение по нейронке
            for (int i = 0; i < inputSignals.Length; i++)
            {
                var signal = new List<double>() { inputSignals[i] };
                var neyron = Layers[0].Neyrons[i];

                neyron.FeedForward(signal);

            }

        }

        private void CreateOutputLayer()
        {
            var outputNeyrons = new List<Neyron>();
            var lastLayer = Layers.Last();
            for (int i = 0; i < Topology.OutputCount; i++)
            {
                var neyron = new Neyron(lastLayer.NeyronCount, NeyronType.Output);
                outputNeyrons.Add(neyron);
            }
            var outputlayer = new Layer(outputNeyrons, NeyronType.Output);
            Layers.Add(outputlayer);
        }

        private void CreateInputLayer()
        {
            var inputNeyrons = new List<Neyron>();
            for (int i = 0; i < Topology.InputCount; i++)
            {
                var neyron = new Neyron(1,NeyronType.Input);
                inputNeyrons.Add(neyron);
            }
            var inputlayer = new Layer(inputNeyrons, NeyronType.Input);
            Layers.Add(inputlayer);
        }

        private void CreateHiddenLayers()
        {
            for (int j = 0; j < Topology.HiddenLayers.Count; j++)
            {
                var hiddenNeyrons = new List<Neyron>();
                var lastLayer = Layers.Last();
                for (int i = 0; i < Topology.HiddenLayers[j]; i++)
                {
                    var neyron = new Neyron(lastLayer.NeyronCount);
                    hiddenNeyrons.Add(neyron);
                }
                var hiddenlayer = new Layer(hiddenNeyrons);
                Layers.Add(hiddenlayer);
            }
        }
    }
}
