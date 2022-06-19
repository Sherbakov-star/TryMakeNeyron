using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeralNetwork
{
    public class Layer
    {
        public List<Neyron> Neyrons { get;}
        public int NeyronCount => Neyrons?.Count ?? 0;

        public NeyronType Type;

        public Layer(List<Neyron> neyrons, NeyronType type = NeyronType.Normal)
        {
            //Проверить все входные нейроны на соответствие типу
            Neyrons = neyrons;
            Type = type;
        }

        public List<double> GetSignals()
        {
            var result = new List<double>();
            foreach(var neyron in Neyrons)
            {
                result.Add(neyron.Output);
            }
            return result;
        }

        public override string ToString()
        {
            return Type.ToString();
        }
    }
}
