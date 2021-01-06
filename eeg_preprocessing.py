import scipy.io
import numpy as np
import torch
import os
import random
import pandas as pd
from scipy.spatial import distance_matrix

from keras.utils import to_categorical

import torch_geometric
#from RGNN import model_simple
#TODO: re-define better the imports, some things may be not needed here


#--------------------------
# Definition of the adjacency matrix
#--------------------------

def global_channels(adjacency_matrix, filtered_channels, channel_tuples):
  for channel1,channel2 in channel_tuples:
    adjacency_matrix[filtered_channels.index(channel1), filtered_channels.index(channel2)] = adjacency_matrix[filtered_channels.index(channel1), filtered_channels.index(channel2)] - 1
    adjacency_matrix[filtered_channels.index(channel2), filtered_channels.index(channel1)] = adjacency_matrix[filtered_channels.index(channel2), filtered_channels.index(channel1)] - 1
  return adjacency_matrix

# TODO: put right folder, maybe pass it as parameter.
def get_adjacency_matrix(path_order, path_location):
  channel_order = pd.read_excel(path_order, header=None)
  channel_location = pd.read_csv(path_location, sep= ",")
  filtered_df = pd.DataFrame(columns=["Channel", "X", "Y", "Z"])
  for channel in channel_location["Channel"]:
    for used in channel_order[0]:
      if channel.upper() == used:
        filtered_df = pd.concat([channel_location.loc[channel_location['Channel'] == channel], filtered_df], ignore_index=True)
  filtered_df = filtered_df.reindex(index=filtered_df.index[::-1]).reset_index(drop=True)
  filtered_matrix = np.asarray(filtered_df.values[:, 1:4], dtype=float)
  distances_matrix = distance_matrix(filtered_matrix, filtered_matrix)
  delta = 5
  adjacency_matrix = np.minimum(np.ones([62,62]), delta/(distances_matrix**2))
  filtered_channels = list(filtered_df["Channel"])
  adjacency_matrix = global_channels(adjacency_matrix, filtered_channels,
    [("Fp1", "Fp2"), ("AF3", "AF4"), ("F5", "F6"), ("FC5", "FC6"), ("C5", "C6"), ("CP5", "CP6"), ("P5", "P6"), ("PO5", "PO6"), ("O1", "O2")]
  )
  adjacency_matrix = np.absolute(adjacency_matrix)
  return adjacency_matrix

#--------------------------
# Emotion-Distribution learning
#--------------------------
def emotionDL(eps=0.1):
  labels = { 
    '1' : to_categorical(np.array([1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3])), 
    '2' : to_categorical(np.array([2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1])), 
    '3' : to_categorical(np.array([1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]))
  }

  for i in range(1, 4):
    for row in labels[str(i)]:
      if (row == [1, 0, 0, 0]).all():
        row[0] = row[0]-3/4*eps
        row[1:4] = row[1:4]+eps/4
      elif (row == [0, 1, 0, 0]).all():
        row[0] = row[0]+1/3*eps
        row[2] = row[2]+1/3*eps
        row[1] = row[1]-2/3*eps
      elif (row == [0, 0, 1, 0]).all():
        row[0] = row[0]+1/4*eps
        row[1] = row[1]+1/4*eps
        row[2] = row[2]-3/4*eps
        row[3] = row[3]+1/4*eps
      elif (row==[0, 0, 0, 1]).all():
        row[0] = row[0]+1/3*eps
        row[2] = row[2]+1/3*eps
        row[3] = row[3]-2/3*eps
  return labels

# TODO: put the labels in a main file, because they depend on the dataset

def standardization(X):
  X = np.array(X)
  return (X-np.mean(X, axis=0))/np.std(X, axis=0)

def normalization(X):
  X = np.array(X)
  return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))

# --- Subject independent ---

def load_data(labels):
  X = []
  Y = []
  P = []
  for session in range (1, 4):
    print("WE ARE IN SESSION: {:d}".format(session))
    session_labels = labels[str(session)]
    session_folder = "/content/data/"+str(session)
    patient_nr = 1
    for patient in os.listdir(session_folder):
      file_path = session_folder+"/"+patient
      print("THIS IS PATIENT: {:d}".format(patient_nr))
      for videoclip in range(24):
        x = scipy.io.loadmat(file_path)['de_LDS{}'.format(videoclip+1)][:,:,:]
        for t in range(x.shape[1]):
          X.append(x[:,t,:])
          Y.append(session_labels[videoclip])
          P.append(patient_nr)
      patient_nr = patient_nr + 1
  return X,Y,P

def create_data_list(X,Y,P):
  data_list = {}
  for s in range(1, 16):
    data_list[str(s)] = []

  for x, y, p in zip(X, Y, P):
    x = torch.tensor(x).float()
    y = torch.tensor([y]).float()
    edge_index = torch.tensor([np.arange(62*62)//62,np.arange(62*62)%62])
    data=torch_geometric.data.Data(x=x,y=y,edge_index=edge_index)
    data_list[str(p)].append(data)
  
  return data_list

def create_data_loaders(data_list,subjects_list,batch_size):
  cross_train_list = {}
  cross_val_loaders = {}

  for s in subjects_list:
    cross_val_loaders[str(s)] = torch_geometric.data.DataLoader(data_list[str(s)], batch_size = batch_size, shuffle=False)

    cross_train_list[str(s)]={}
    cross_train_list[str(s)]["data"]=[]
    for z in subjects_list:
      if s!=z:
        cross_train_list[str(s)]["data"].extend(data_list[str(z)])
  # check that all of this is correct
  for s in range(1, 16):
    random.shuffle(cross_train_list[str(s)]["data"])
    cross_train_list[str(s)]["domain_train_loader"] = torch_geometric.data.DataLoader(
        cross_train_list[str(s)]["data"][:len(cross_train_list[str(s)]["data"])//2],
        batch_size = batch_size, shuffle=False
    )
    
    cross_train_list[str(s)]["domain_val_loader"] = torch_geometric.data.DataLoader(
        cross_train_list[str(s)]["data"][len(cross_train_list[str(s)]["data"])//2:],
        batch_size = batch_size, shuffle=False
    )
  return cross_train_list,cross_val_loaders

# --- Subject dependent ---

def load_data_subjects(labels):
  eeg_dict = {}
  for session in range (1, 4):
    print("WE ARE IN SESSION: {:d}".format(session))
    session_labels = labels[str(session)]
    session_folder = "/content/data/"+str(session)
    for file in os.listdir(session_folder):
      p=int(file.split("_")[0])
      file_path = "{}/{}".format(session_folder,file)
      print("THIS IS PATIENT: {:d}".format(p))
      for videoclip in range(24):
        x = scipy.io.loadmat(file_path)['de_LDS{}'.format(videoclip+1)]
        if p not in eeg_dict.keys():
          eeg_dict[p] = {
            "neutral": {"train": [],"val": []},
            "sadness": {"train": [],"val": []},
            "fear": {"train": [],"val": []},
            "happiness": {"train": [],"val": []}
          }
        y=session_labels[videoclip]
        emotion=list(eeg_dict[p].keys())[np.argmax(y)]
        data=(x,y)
        if (len(eeg_dict[p][emotion]["train"]) - (session-1)*4) <4:
          eeg_dict[p][emotion]["train"].append(data)
        else:
          eeg_dict[p][emotion]["val"].append(data)
  eeg_dict_no_emo={}
  for p in eeg_dict.keys():
    eeg_dict_no_emo[p]={"train": [],"val": []}
    for phase in ["train","val"]:
      data=[]
      for emotion in eeg_dict[p].keys():
        for X,y in eeg_dict[p][emotion][phase]:
          y = torch.tensor([y]).float()
          edge_index = torch.tensor([np.arange(62*62)//62, np.arange(62*62)%62])
          for t in range(X.shape[1]):
            x = torch.tensor(X[:, t, :]).float()
            data.append(torch_geometric.data.Data(x=x, y=y, edge_index=edge_index))
      eeg_dict_no_emo[p][phase]=data
  return eeg_dict_no_emo

def create_data_loaders_subjects(list_train, list_val, batch_size):
  loader_train = {}
  loader_val = {}
  for i in range(1, 16):
    loader_train[str(i)] = torch_geometric.data.DataLoader(list_train[str(i)], batch_size=batch_size, shuffle=False)
    loader_val[str(i)] = torch_geometric.data.DataLoader(list_val[str(i)], batch_size=batch_size, shuffle=False)
  return loader_train, loader_val
