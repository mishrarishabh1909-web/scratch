import networkx as nx
from yafs.topology import Topology
import yaml

class TopologyBuilder:
    def __init__(self, config_path="config.yaml"):
        # For simplicity, fallback to default values if config isn't loaded fully
        self.num_fog_nodes = 5
        self.num_edge_devices = 20
        self.cloud_ipt = 10000
        self.cloud_ram = 32000
        self.fog_ipt = 1000
        self.fog_ram = 8000
        
    def build(self):
        G = nx.Graph()
        
        # Cloud Node (ID: 0)
        G.add_node(0, type='cloud', IPT=self.cloud_ipt, RAM=self.cloud_ram)
        
        # Fog Nodes (IDs: 1 to num_fog_nodes)
        for i in range(1, self.num_fog_nodes + 1):
            G.add_node(i, type='fog', IPT=self.fog_ipt, RAM=self.fog_ram)
            # High latency (100ms), High bandwidth to cloud
            G.add_edge(i, 0, PR=100, BW=1000)
            
        # Edge Devices (IDs: num_fog_nodes+1 to num_fog_nodes+num_edge_devices)
        start_edge = self.num_fog_nodes + 1
        for i in range(start_edge, start_edge + self.num_edge_devices):
            G.add_node(i, type='edge', IPT=0, RAM=0)
            # Low latency (10ms), Medium bandwidth to a specific fog node
            target_fog = 1 + (i % self.num_fog_nodes)
            G.add_edge(i, target_fog, PR=10, BW=100)
            
        topology = Topology()
        topology.G = G
        return topology
