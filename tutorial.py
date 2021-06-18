import torch
from torch_geometric.data import Data

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader

#
# edge_index = torch.tensor([[0, 1, 1, 2],
#                            [1, 0, 2, 1]], dtype=torch.long)
# x = torch.tensor([[-1], [0], [1]], dtype=torch.float)


# Equivalente
edge_index = torch.tensor([[0, 1],
                           [1, 0],
                           [1, 2],
                           [2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)

data = Data(x=x, edge_index=edge_index.t().contiguous())

# data = Data(x=x, edge_index=edge_index)

for key, item in data:
    print("{} found in data".format(key))


num_nodes = data.num_nodes
num_edges = data.num_edges
num_node_features = data.num_node_features
isolated_nodes =  data.contains_isolated_nodes
data.contains_self_loops()
data.is_directed()
device = torch.device('cuda')
data = data.to(device)

dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
loader = DataLoader(dataset, batch_size=2, shuffle=True)
len(dataset)
dataset[0]
for batch in loader:
    batch
