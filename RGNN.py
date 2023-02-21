# PyTorch dependencies
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
import matplotlib.pyplot as plt
from torch_geometric.nn import SGConv, global_add_pool
import torch_geometric
from torch_scatter import scatter_add
from tensorflow.python.client import device_lib
# Std dependencies
import json
import os
import random
import math
# Support math dependencies
import pandas as pd
import numpy as np
from scipy.spatial import distance_matrix
import scipy.io

### Original proposed code:
#######################################################################################
# HELPER FUNCTIONS

# maybe_num_nodes: derives the correct number of nodes using the edges, if not given
#
# Parameters:
# - index: Tensor (2, num_edges), see next function
# - num_nodes: number of nodes, if known; it will be otherwise computed from the indices
def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes   # If num_nodes == None, find num_nodes as the max number among the edge indices (+1, as they start from 0)


# add_remaining_self_loops: takes an adjacency matrix in the form of edge_index and edge_weight lists, and adds self loops if not present
#
# Parameters:
# - edge_index: Tensor (2, num_edges), where each col i represent the i-th edge (x, y)
# - edge_weight: Tensor (1, num_edges), which contains the weight of the i-th edge for each col i
# - fill_value=1: value to fill the diagonal with
# - num_nodes: number of nodes present, if known (it will be derived otherwise)
def add_remaining_self_loops(edge_index, edge_weight=None, fill_value=1, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)  # Derive num_nodes with "maybe_num_nodes" if unknown
    row, col = edge_index   # obtain the two rows of edge_index as two separate tensors, one of edge rows (x's) and the other of edge cols (y's)
    row = row.long()
    mask = row != col   # a bit-wise mask of all matrix cells, excluding the diagonal
    inv_mask = torch.logical_not(mask) # inverted mask, that is ONLY the diagonal
    loop_weight = torch.full((num_nodes, ), fill_value, dtype=None if edge_weight is None else edge_weight.dtype, device=edge_index.device)
    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)  # Assert that the number of edge_weights is the same as the number of edge_indices couples
        remaining_edge_weight = edge_weight[inv_mask]   # Subset for the edge_weights of edges on the diagonal
        if remaining_edge_weight.numel() > 0:                      # If there was already at least an edge_weight on the diagonal:
            loop_weight[row[inv_mask]] = remaining_edge_weight  # makes it so loop_weight[i] represents the weight on the cell (i,i); sets it to the currently available value, and leaves it at 1 if not present
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0) # Updates edge_weights as concat([ edge_weight[not_diagonal], edge_weight[diagonal] ]), where the edge weights on the diagonal are currently inside loop_weight

    # Repeats the same steps to update and reorder the edge_index values as concat( [ edge_index[not_diagonal], edge_index[diagonal] ] ), by using a tempo loop_index tensor
    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


#######################################################################################
# HELPER CLASSES

# NewSGConv(SGConv): super-class of SGConv with some redefinitions to allow negative values in the adjacency matrix
class NewSGConv(SGConv):  # SGConv implements the base SGC computations, such as S=D^(-1/2)AD^(-1/2) and A'=A+I for the self-loops
    def __init__(self, num_features, num_classes, K=1, cached=False,
                 bias=True):
        super(NewSGConv, self).__init__(num_features, num_classes, K=K, cached=cached, bias=bias)

    # N.B.: SGConv implements the adjacency matrix via two separates structures: a list of indices (x, y) of the edges ("edge_index") and a list of their assigned weights ("edge_weight")

    # allow negative edge weights
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None: # Uses a default edge_weight initialization, if not provided
            edge_weight = torch.ones((edge_index.size(1), ),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops( # adds the remaining self-loops, if necessary
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index

        # N.B.: scatter_add creates a Tensor (num_nodes, 1) "deg" where "deg[i] = sum(A_i, i=0) -> puts in deg[i] the sum of row i of A
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes) # torch.abs is required because A contains negative values, see chap 4.2.0 in the paper
        deg_inv_sqrt = deg.pow(-0.5)  # computes D^(-1/2)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0  # deletes "inf" values

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]  # Returns  S=D^(-1/2)AD^(-1/2), optimized by using a 1-dimensional tensor instead of 2D diagonal matrices

# N.B. "forward" is a redefinition of SGConv's forward method, done to use the here defined "norm" method for normalization, in substitution of the default "gcn_norm"
    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = NewSGConv.norm(  # Returns normalized edge_index and edge_weights
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):   # see chap 3.1.0, fig.4; we assume K=2
                x = self.propagate(edge_index, x=x, norm=norm)
            self.cached_result = x

        return self.lin(self.cached_result)

    def message(self, x_j, norm): # called by "self.propagate"
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j

# ReverseLayerF(Function): extends Function, is used in the NodeDAT, both "forward" and "backward" methods are provided for activation and backward propagation
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

#######################################################################################
# THE RGNN MODEL

class SymSimGCNNet(torch.nn.Module):
    def __init__(self, num_nodes, learn_edge_weight, edge_weight, num_features, num_hiddens, num_classes, K, dropout=0.5, domain_adaptation=""):
        """
            num_nodes: number of nodes in the graph
            learn_edge_weight: if True, the edge_weight is learnable
            edge_weight: initial edge matrix
            num_features: feature dim for each node/channel
            num_hiddens: a tuple of hidden dimensions                     # d'
            num_classes: number of emotion classes
            K: number of layers                                                           # L=2
            dropout: dropout rate in final linear layer                           # 0.7
            domain_adaptation: RevGrad
        """
        super(SymSimGCNNet, self).__init__()
        self.domain_adaptation = domain_adaptation
        self.num_nodes = num_nodes
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0) # cols and rows of edge_index
        edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[self.xs, self.ys] # strict lower triangular values
        self.edge_weight = nn.Parameter(edge_weight, requires_grad=learn_edge_weight) # edge_weight as lower triangular is set as a learnable parameter
        self.dropout = dropout
        # LAYERS
        self.conv1 = NewSGConv(num_features=num_features, num_classes=num_hiddens[0], K=K)  # a K-layer SGC
        self.fc = nn.Linear(num_hiddens[0], num_classes)  # an output linear classification layer [d' * C] 
        # NODE-DAT
        if self.domain_adaptation in ["RevGrad"]:
            self.domain_classifier = nn.Linear(num_hiddens[0], 2) # also creates an output layer [d' * 2] for the NodeDAT

        self.apply(self.weight_init) # apply the following function

    # function for the Xavier initialization of the weights with uniform distribution
    def weight_init(self,m):
      if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
      if isinstance(m, NewSGConv):
        nn.init.xavier_uniform_(m.lin.weight)

    def forward(self, data, alpha=0):
        # "data" created using the Data class, see: https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs
        # "alpha" corresponds to the paper's Beta parameter in the NodeDAT adversarial training
        batch_size = len(data.y)
        x, edge_index = data.x, data.edge_index

        # Create a local copy of the edge_weight, as upper triangular
        edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
        edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
        edge_weight = edge_weight + edge_weight.transpose(1,0) - torch.diag(edge_weight.diagonal()) # copy values from lower tri to upper tri

        # Repeat edge_weight batch_size times
        edge_weight = edge_weight.reshape(-1).repeat(batch_size)
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        
        # domain classification
        domain_output = None
        if self.domain_adaptation in ["RevGrad"]:
            reverse_x = ReverseLayerF.apply(x, alpha) # forward step in ReverseLayerF
            domain_output = self.domain_classifier(reverse_x)
        x = global_add_pool(x, data.batch, size=batch_size) # Node-wise sum pooling
        x = F.dropout(x, p=self.dropout, training=self.training) # Dropout
        x = self.fc(x)  # Linear output
        return x, domain_output

#######################################################################################
# Functions for creating a correctly initialized adjacency matrix
def global_channels(adjacency_matrix, filtered_channels, channel_tuples):
  for channel1,channel2 in channel_tuples:  # Update the global channel values as "value = value - 1"
    adjacency_matrix[filtered_channels.index(channel1), filtered_channels.index(channel2)] = adjacency_matrix[filtered_channels.index(channel1), filtered_channels.index(channel2)] - 1
    adjacency_matrix[filtered_channels.index(channel2), filtered_channels.index(channel1)] = adjacency_matrix[filtered_channels.index(channel2), filtered_channels.index(channel1)] - 1
  return adjacency_matrix

def get_adjacency_matrix():
  channel_order = pd.read_excel(root_dir+"Channel Order.xlsx", header=None) # Use the Excel sheet for indexing, as the "channel_locations.txt" file contains extra rows
  channel_location = pd.read_csv(root_dir+"channel_locations.txt", sep= "\t", header=None)
  channel_location.columns = ["Channel", "X", "Y", "Z"]
  channel_location["Channel"] = channel_location["Channel"].apply(lambda x: x.strip().upper())
  filtered_df = pd.DataFrame(columns=["Channel", "X", "Y", "Z"])
  for channel in channel_location["Channel"]:
    for used in channel_order[0]:
      if channel == used:
        filtered_df = pd.concat([channel_location.loc[channel_location['Channel'] == channel], filtered_df], ignore_index=True) # Concatenate each row from the "channel_location" dataframe whose channel name is present in the Excel sheet
  filtered_df = filtered_df.reindex(index=filtered_df.index[::-1]).reset_index(drop=True)
  filtered_matrix = np.asarray(filtered_df.values[:, 1:4], dtype=float)
  distances_matrix = distance_matrix(filtered_matrix, filtered_matrix)  # Compute a matrix of distances for each combination of channels in the filtered_matrix dataframe
  delta = 5
  adjacency_matrix = np.minimum(np.ones([62,62]), delta/(distances_matrix**2))  # Computes the adjacency matrix cells as min(1, delta/d_ij), N.B. zero division error can arise here for the diagonal cells, "1" value will be chosen automatically instead
  filtered_channels = list(filtered_df["Channel"])
  adjacency_matrix = global_channels(adjacency_matrix, filtered_channels,
    [("FP1", "FP2"), ("AF3", "AF4"), ("F5", "F6"), ("FC5", "FC6"), ("C5", "C6"), ("CP5", "CP6"), ("P5", "P6"), ("PO5", "PO6"), ("O1", "O2")]  # These couples are taken from the paper, chap. 4.1
  )
  return torch.tensor(adjacency_matrix).float()

#######################################################################################
# Functions to create dataloaders from the SEED-IV dataset

# --- Subject dependent ---
def dependent_loaders(batch_size, shuffle, root_dir):
  session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
  session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
  session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];
  labels = [session1_label, session2_label, session3_label]
  eeg_dict = {}

  print("--- SUBJECT DEPENDENT LOADER ---")
  
  for session in range (0, 3):
    print("WE ARE IN SESSION: {:d}".format(session + 1))
    session_labels = labels[session]
    session_folder = root_dir + "eeg_feature_smooth/" + str(session + 1)

    for file in os.listdir(session_folder):
      subject_number = int(file.split("_")[0])
      file_path = "{}/{}".format(session_folder,file)
      print("THIS IS PATIENT: {:d}".format(subject_number))
      
      # Create a dict containing a "train" Data objects List and a "test" Data objects List, for each subject
      if subject_number not in eeg_dict.keys():
          eeg_dict[subject_number] = {"train": [], "test": []}

      # Each subject has 24 trails
      for trial in range(24): 
        X = scipy.io.loadmat(file_path)['de_LDS{}'.format(trial + 1)]
        y = session_labels[trial]
        y = torch.tensor([y]).long()
        edge_index = torch.tensor([np.arange(62*62)//62, np.arange(62*62)%62])

        for t in range(X.shape[1]):
          x = torch.tensor(X[:, t, :]).float()
          x = (x-x.mean())/x.std()
          if trial < 16:  # Put the first 16 in training, and the last 8 in testing
            eeg_dict[subject_number]["train"].append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))
          else:
            eeg_dict[subject_number]["test"].append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))
  
  loaders_dict = {}

  # Adapt the "train" and "test" Data lists into single DataLoader objects
  for key in eeg_dict.keys():
    loaders_dict[key] = {}
    loaders_dict[key]["train"] = torch_geometric.data.DataLoader(eeg_dict[key]["train"], batch_size=batch_size, shuffle=shuffle)
    loaders_dict[key]["test"] = torch_geometric.data.DataLoader(eeg_dict[key]["test"], batch_size=batch_size, shuffle=shuffle)

  print("--------------------------------\n\n")
  return loaders_dict


# --- Subject independent ---
def independent_loaders(batch_size, shuffle, root_dir):
  session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3];
  session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1];
  session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0];
  labels = [session1_label, session2_label, session3_label]
  eeg_dict = {}
  print("--- SUBJECT INDEPENDENT LOADER ---")
  for session in range (1, 4):
    print("WE ARE IN SESSION: {:d}".format(session))
    session_labels = labels[session-1]
    session_folder = root_dir+"eeg_feature_smooth/"+str(session)
    for file in os.listdir(session_folder):
      subject_number = int(file.split("_")[0])
      file_path = "{}/{}".format(session_folder,file)
      print("THIS IS PATIENT: {:d}".format(subject_number))
      # Objective: create for each subject a single list of Data object (all samples of a subject are used as either training or testing)
      if subject_number not in eeg_dict.keys(): eeg_dict[subject_number] = []

      for trial in range(24): # There are 24 trials per subject, following the "session_label" label order
        X = scipy.io.loadmat(file_path)['de_LDS{}'.format(trial+1)]
        y = session_labels[trial]
        y = torch.tensor([y]).long()
        edge_index = torch.tensor([np.arange(62*62)//62, np.arange(62*62)%62])
        for t in range(X.shape[1]):
          x = torch.tensor(X[:, t, :]).float()
          x = (x-x.mean())/x.std()
          eeg_dict[subject_number].append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))
  
  loaders_dict = {}

  # Keep the bare list in a "list" field, and adapt it into a single DataLoader object in a "loader" field
  for key in eeg_dict.keys():
    loaders_dict[key] = {}
    loaders_dict[key]["list"] = eeg_dict[key]
    loaders_dict[key]["loader"] = torch_geometric.data.DataLoader(eeg_dict[key], batch_size=batch_size, shuffle=shuffle)

  print("----------------------------------\n\n")
  return loaders_dict

#######################################################################################
# Training functions, return a dictionary of per-patient losses and accuracy over the epochs

#--- Subject Dependent ---
def subject_dependent_training(dep_loader, emo_DL=True, rand_adj=False, L1_alpha=0.01, noise_level=0.2, num_hiddens=16, num_epochs=100):
  model_stats = {}
  nodeDAT = ""
  eps3 = noise_level/3
  eps4 = noise_level/4
  emoDL_map = torch.tensor([
      [1-3*eps4, eps4, eps4, eps4],
      [eps3, 1-2*eps3, eps3, 0],
      [eps4, eps4, 1-3*eps4,eps4],
      [eps3, 0, eps3, 1-2*eps3]
  ])

  for i in range(1, 16):
    print("Patient no. "+str(i)+":")
    loss_history = {'train': [], 'test': []}
    accuracy_history = {'train': [], 'test': []}

    edge_weight = get_adjacency_matrix() if not rand_adj else torch.rand((62,62))
    
    model = SymSimGCNNet(62, True, edge_weight, 5, [num_hiddens], 4, 2, dropout=0.7, domain_adaptation=nodeDAT)
    model = model.to(dev)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    data_loaders = dep_loader[i]
    for epoch in range(num_epochs):
      # Initialize variables for computing average loss/accuracy
      # Initialize epoch variables
      sum_loss = {"train": 0, "test": 0}
      sum_accuracy = {"train": 0, "test": 0}
      # Process each split
      for split in ["train", "test"]:
          # Set network mode (train or eval)
          if split == "train":
            model.train()
            torch.set_grad_enabled(True)
          else:
            model.eval()
            torch.set_grad_enabled(False)
          # Process all data in split
          for batch in data_loaders[split]:
            # Move to CUDA
            batch = batch.to(dev)
            # Reset gradients
            optimizer.zero_grad()
            # Compute output
            pred,_ = model(batch)
            y_list = batch.y.tolist()
            
            loss = None
            if emo_DL:
              target = emoDL_map[y_list].to(dev)
              loss = F.kl_div(F.log_softmax(pred, -1), target, reduction="sum") + L1_alpha * torch.norm(model.edge_weight) # w/ emotionDL
            else:
              loss = -F.log_softmax(pred, -1)[range(0,len(y_list)), y_list] + L1_alpha * torch.norm(model.edge_weight) # w/out emotionDL
              loss = torch.sum(loss)
            if nodeDAT in ["RevGrad"]:
              pass
            # Update loss
            sum_loss[split] += loss.item()
            # Check parameter update
            if split == "train":
                # Compute gradients
                loss.backward()
                # Optimize
                optimizer.step()
            # Compute accuracy
            _,pred_target = pred.max(1)
            batch_accuracy = (pred_target == batch.y).sum().item()/batch.y.size(0)
            # Update accuracy
            sum_accuracy[split] += batch_accuracy
      # Compute average epoch loss/accuracy
      epoch_loss = {split: sum_loss[split]/len(data_loaders[split]) for split in ["train", "test"]}
      epoch_accuracy = {split: sum_accuracy[split]/len(data_loaders[split]) for split in ["train", "test"]}
      # Update history
      for split in ["train", "test"]:
        loss_history[split].append(epoch_loss[split])
        accuracy_history[split].append(epoch_accuracy[split])
      # Print info
      epoch = epoch + 1
      if epoch==1 or (epoch%10)==0:
        print(f"Epoch {epoch}:",
              f"TrL={epoch_loss['train']:.4f},",
              f"TrA={epoch_accuracy['train']:.4f},",
              f"TeL={epoch_loss['test']:.4f},",
              f"TeA={epoch_accuracy['test']:.4f},")
    print("\n\n")
    model_stats[i] = {
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
        "best_losses": {"train": loss_history["train"][-1], "test": loss_history["test"][-1]},
        "best_accuracies":{"train":accuracy_history["train"][-1], "test": accuracy_history["test"][-1]}
    }
    print(str(i))
    print(model_stats[i]["best_losses"]['train'])
    break
  return model_stats

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--- Subject Independent ---
def subject_independent_training(indep_loader, nodeDAT="RevGrad", emo_DL=True, rand_adj=False, L1_alpha=0.01, noise_level=0.2, num_hiddens=16, num_epochs=100):
  model_stats = {}
  eps3 = noise_level/3
  eps4 = noise_level/4
  emoDL_map = torch.tensor([
      [1-3*eps4, eps4, eps4, eps4],
      [eps3, 1-2*eps3, eps3, 0],
      [eps4, eps4, 1-3*eps4,eps4],
      [eps3, 0, eps3, 1-2*eps3]
  ])

  for i in range(1,16): 
    print("PAZIENTE no. "+str(i)+" (cross-validation):")
    indices = [j for j in range(1, 16) if j!=i] # Train on all but i, leave i for last for testing
    indices.append(i)

    num_samples = len(indep_loader[i]["list"])

    loss_history = {'train': [], 'test': []}
    accuracy_history = {'train': [], 'test': []}

    edge_weight = get_adjacency_matrix() if not rand_adj else torch.rand((62,62))
    
    model = SymSimGCNNet(62, True, edge_weight, 5, [num_hiddens], 4, 2, dropout=0.7, domain_adaptation=nodeDAT)
    model = model.to(dev)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(num_epochs):
      # Initialize variables for computing average loss/accuracy
      # Initialize epoch variables
      sum_loss = {"train": 0, "test": 0}
      sum_accuracy = {"train": 0, "test": 0}
      count = {"train": 0, "test": 0}

      # Compute the GRL beta factor
      p = epoch/num_epochs
      beta = 2/(1+math.exp(-10*p))-1

      for ind in indices:
          data_loader = indep_loader[ind]["loader"]
          split = "train" if ind!=i else "test"
          # Set network mode (train or eval)
          if split == "train":
            model.train()
            torch.set_grad_enabled(True)
          else:
            model.eval()
            torch.set_grad_enabled(False)
          # Process all data in split
          for batch in data_loader:
            # Move to CUDA
            batch = batch.to(dev)
            # Reset gradients
            optimizer.zero_grad()
            # Compute output
            pred, source_dom = model(batch, beta)
            y_list = batch.y.tolist()
            batch_size = len(y_list)
            
            loss = None
            # EmoDL:
            if emo_DL:
              target = emoDL_map[y_list].to(dev)
              loss = F.kl_div(F.log_softmax(pred, -1), target, reduction="sum") + L1_alpha * torch.norm(model.edge_weight) # w/ emotionDL
            else:
              loss = -F.log_softmax(pred, -1)[range(0,batch_size), y_list] + L1_alpha * torch.norm(model.edge_weight) # w/out emotionDL
              loss = torch.sum(loss)
            # NodeDAT
            if nodeDAT in ["RevGrad"]:
              random_pick = [n for n in range(num_samples)]
              random.shuffle(random_pick)
              random_pick = random_pick[:batch_size]
              target_loader = torch_geometric.data.DataLoader([indep_loader[i]["list"][ra] for ra in random_pick], batch_size=batch_size, shuffle=True)
              target_batch = next(iter(target_loader))
              target_batch = target_batch.to(dev)
              _, target_dom = model(target_batch, beta)
              NodeDatLoss = -F.log_softmax(source_dom, -1)[range(0,batch_size*62), [0 for i in range(batch_size*62)]] -F.log_softmax(target_dom, -1)[range(0,batch_size*62), [1 for i in range(batch_size*62)]]
              NodeDatLoss = torch.sum(loss)
              loss = loss + NodeDatLoss
            # Update loss
            sum_loss[split] += loss.item()
            # Check parameter update
            if split == "train":
                # Compute gradients
                loss.backward()
                # Optimize
                optimizer.step()
            # Compute accuracy
            _,pred_target = pred.max(1)
            batch_accuracy = (pred_target == batch.y).sum().item()/batch.y.size(0)
            # Update accuracy
            sum_accuracy[split] += batch_accuracy

            count[split] = count[split] + 1
            
      # Compute average epoch loss/accuracy
      epoch_loss = {split: sum_loss[split]/count[split] for split in ["train", "test"]}
      epoch_accuracy = {split: sum_accuracy[split]/count[split] for split in ["train", "test"]}
      # Update history
      for split in ["train", "test"]:
        loss_history[split].append(epoch_loss[split])
        accuracy_history[split].append(epoch_accuracy[split])
      # Print info
      epoch = epoch + 1
      if epoch==1 or (epoch%5)==0:
        print(f"Epoch {epoch}:",
              f"TrL={epoch_loss['train']:.4f},",
              f"TrA={epoch_accuracy['train']:.4f},",
              f"TeL={epoch_loss['test']:.4f},",
              f"TeA={epoch_accuracy['test']:.4f},")
    print("\n\n")
    model_stats[i] = {
        "loss_history": loss_history,
        "accuracy_history": accuracy_history,
        "best_losses": {"train": loss_history["train"][-1], "test": loss_history["test"][-1]},
        "best_accuracies":{"train":accuracy_history["train"][-1], "test": accuracy_history["test"][-1]}
    }
    print(str(i))
    print(model_stats[i]["best_losses"]['train'])
  return model_stats

#######################################################################################
# Plotting functions

def graph_res(i, res):
  loss_history = res[i]["loss_history"]
  accuracy_history = res[i]["accuracy_history"]

  # Plot loss history
  plt.title("Loss")
  for split in ["train", "test"]:
    plt.plot(loss_history[split][3:], label=split)  # Discard the first 3 epochs
  plt.legend()
  plt.show()

  # Plot accuracy history
  plt.title("Accuracy")
  for split in ["train", "test"]:
    plt.plot(accuracy_history[split][3:], label=split)  # Discard the first 3 epochs
  plt.legend()
  plt.show()

def show_results(res):
  for i in range(1,16):
    print(i)
    print(res[i]["best_losses"]['train'])
    graph_res(i, res)
    print("\n\n\n")

#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################
#######################################################################################

if __name__ == "__main__":
    # Correctly setup the device
    dev = ("cuda" if torch.cuda.is_available() else "cpu")

    # from google.colab import drive  # Connect to Google Drive instead of local dataset
    # drive.mount("/content/drive", force_remount=True)
    root_dir = "./SEED-IV/"       # Change this line to use a local root directory

    # Create subject-dependent and independent loaders, with batch_size=16 and shuffle
    dep_loader = dependent_loaders(16, True, root_dir)
    indep_loader = independent_loaders(16, True, root_dir)

    ## SUBJECT-DEPENDENT training and result plotting
    subject_dependent_result = subject_dependent_training(dep_loader)
    show_results(subject_dependent_result)

    ## SUBJECT-INDEPENDENT training and result plotting
    subject_independent_result = subject_independent_training(indep_loader)
    show_results(subject_independent_result)