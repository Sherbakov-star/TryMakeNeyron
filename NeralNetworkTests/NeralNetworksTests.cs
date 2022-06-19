using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeralNetwork;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeralNetwork.Tests
{
    [TestClass()]
    public class NeralNetworksTests
    {
        [TestMethod()]
        public void FeedForwardTest()
        {
            var outputs = new double[] { 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1 };
            var inputs = new double[,]
            {
                // Результат - Пациент болен - 1
                //             Пациент Здоров - 0

                // Неправильная температура T
                // Хороший возраст A
                // Курит S
                // Правильно питается F
                //T  A  S  F
                { 0, 0, 0, 0 },
                { 0, 0, 0, 1 },
                { 0, 0, 1, 0 },
                { 0, 0, 1, 1 },
                { 0, 1, 0, 0 },
                { 0, 1, 0, 1 },
                { 0, 1, 1, 0 },
                { 0, 1, 1, 1 },
                { 1, 0, 0, 0 },
                { 1, 0, 0, 1 },
                { 1, 0, 1, 0 },
                { 1, 0, 1, 1 },
                { 1, 1, 0, 0 },
                { 1, 1, 0, 1 },
                { 1, 1, 1, 0 },
                { 1, 1, 1, 1 }
            };
            var topology = new Topology(4,1,0.1,2);
            var neyralNetwork = new NeralNetworks(topology);

            var difference = neyralNetwork.Learn(outputs,inputs, 10000);//обучение
            var results = new List<double>();
            for (int i = 0; i < outputs.Length; i++)
            {
                var row = NeralNetworks.GetRow(inputs, i);
                var res = neyralNetwork.FeedForward(row).Output;
                results.Add(res);
            }
            

            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Assert.AreEqual(expected, actual);
            }

        }

        [TestMethod()]
        public void DataSetMethod()
        {
            var outputs = new List<double>();
            var inputs = new List<double[]>();
            using (var sr = new StreamReader("nazvanie.csv"))
            {
                var header = sr.ReadLine();

                while (!sr.EndOfStream)
                {
                    var row = sr.ReadLine();
                    var values = row.Split(',').Select(v => Convert.ToDouble(v.Replace("", ""))).ToList(); ;
                    var output = values.Last();
                    var input = values.Take(values.Count - 1).ToArray();

                    outputs.Add(output);
                }
            }
            var inputSignals = new double[inputs.Count, inputs[0].Length];
            for (int i = 0; i < inputSignals.GetLength(0); i++)
            {
                for(var j = 0; j < inputSignals.GetLength(1); j++)
                {
                    inputSignals[i, j] = inputs[i][j];
                }
            }

            var topology = new Topology(outputs.Count, 1, 0.1, outputs.Count / 2);
            var neyralNetwork = new NeralNetworks(topology);
            var difference = neyralNetwork.Learn(outputs.ToArray(), inputSignals, 1000);//обучение

            var results = new List<double>();
            for (int i = 0; i < outputs.Count; i++)
            {
                var res = neyralNetwork.FeedForward(inputs[i]).Output;
                results.Add(res);
            }


            for (int i = 0; i < results.Count; i++)
            {
                var expected = Math.Round(outputs[i], 2);
                var actual = Math.Round(results[i], 2);
                Assert.AreEqual(expected, actual);
            }
        }

        public void RecognizeImages()
        {
            var size = 10000;
            var dataset1Path = @"D:\dataset_for_hakaton\Test\PrivateTestSet\PrivateTestSet\queries";
            var dataset2Path = @"D:\dataset_for_hakaton\Test\PrivateTestSet\PrivateTestSet\shelves";

            var converter = new PictureConvertercs();
            var testImageInput = converter.Convert(@"D:\dataset_for_hakaton\Test\PrivateTestSet\PrivateTestSet\queries\.png");
            var testImageInput2 = converter.Convert(@"D:\dataset_for_hakaton\Test\PrivateTestSet\PrivateTestSet\queries\.png");

            var topology = new Topology(testImageInput.Count, 1, 0.1, testImageInput.Count / 2);
            var neyralNetwork = new NeralNetworks(topology);

            double[,] imagesInputs = GetData(dataset1Path, converter, testImageInput,size);
            neyralNetwork.Learn(new double[] { 1 }, imagesInputs, 1);

            double[,] imagesInputs2 = GetData(dataset2Path, converter, testImageInput,size);
            neyralNetwork.Learn(new double[] { 0 }, imagesInputs2, 1);

            var im1 = neyralNetwork.FeedForward(testImageInput.Select(t => (double)t).ToArray());
            var im2 = neyralNetwork.FeedForward(testImageInput2.Select(t => (double)t).ToArray());

            Assert.AreEqual(1, Math.Round(im1.Output, 2));
            Assert.AreEqual(0, Math.Round(im2.Output, 2));
        }


        private static double[,] GetData(string dataset1Path, PictureConvertercs converter, List<int> testImageInput,int size)
        {
            var images = Directory.GetFiles(dataset1Path);

            var result = new double[size, testImageInput.Count];
            for (int i = 0; i < size; i++)
            {
                var image = converter.Convert(images[i]);
                for (int j = 0; j < image.Count; j++)
                {
                    result[i,j] = image[j];
                }
            }

            return result;
        }
    }
}