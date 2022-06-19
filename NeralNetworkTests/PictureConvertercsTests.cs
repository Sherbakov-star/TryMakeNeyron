using Microsoft.VisualStudio.TestTools.UnitTesting;
using NeralNetwork;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeralNetwork.Tests
{
    [TestClass()]
    public class PictureConvertercsTests
    {
        [TestMethod()]
        public void ConvertTest()
        {
            var converter = new PictureConvertercs();
            var inputs = converter.Convert(@"D:\dataset_for_hakaton\Test\PrivateTestSet\PrivateTestSet\queries\.png");
            converter.Save("D:\\dataset_for_hakaton\\test2\\image.png", inputs);
        }
    }
}