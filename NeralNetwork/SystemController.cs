using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeralNetwork
{
    public class SystemController
    {
        public NeralNetworks DataNetwork { get; }
        public NeralNetworks ImageNetwork { get; }

        public SystemController()
        {
            var imageTopology = new Topology(400, 1, 0.1, 200);
            ImageNetwork = new NeralNetworks(imageTopology);
        }
    }
}
