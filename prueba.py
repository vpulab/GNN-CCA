from torch_geometric.datasets import Planetoid
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from sklearn import metrics
graph = [[ 0, 1 , 1, 0 , 0 ],
[ 0, 0 , 1 , 0 ,0 ],
[ 0, 0, 0, 0, 0],
[0, 0 , 0, 0, 1],
[0, 0, 0, 0, 0]]



graph = [[ 0, 1 , 0, 0 , 0 ],
[ 1, 0 , 0 , 0 ,0 ],
[ 0, 0, 0, 0, 0],
[0, 0 , 0, 0, 1],
[0, 0, 0, 1, 0]]
#
# graph = [[ 0, 0 , 1, 0 , 1 ],
# [ 0, 0 , 0 , 1 ,0 ],
# [ 1, 0, 0, 0, 1],
# [0, 1 , 0, 0, 0],
# [1, 0, 1, 0, 0]]
#
# graph = csr_matrix(graph)
# print(graph)
#
# n_components, labels = connected_components(csgraph=graph, directed=False, return_labels=True)
# print('n components '+  str(n_components))
# print(labels)
labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
labels_pred = [0, 0, 1, 1, 1, 0, 2, 2, 2]

metrics.adjusted_rand_score(labels_true, labels_pred)
