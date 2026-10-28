

import os
from origin_runtime import get_project_dir, get_output_dir, get_data_dir


import pandas as pd


import numpy as np


import re


import pickle


import sys


import time


import json


import gzip


import logging


project_dir = get_project_dir()


def pickle_save_obj(obj, name,dir_path=None):


    if(dir_path==None):


        with open(name + '.pkl', 'wb') as f:


            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


    else:


        with open(dir_path+name + '.pkl', 'wb') as f:


            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def pickle_load_obj(name,dir_path=None):


    if(dir_path==None):


        with open(name + '.pkl', 'rb') as f:


            return pickle.load(f)


    else:


        with open(dir_path+name + '.pkl', 'rb') as f:


            return pickle.load(f)


import torch


import numpy as np


import random


def set_seed(seed=42):


    random.seed(seed)


    torch.manual_seed(seed)


    os.environ['PYTHONHASHSEED'] = str(seed)


    if torch.cuda.is_available():


        torch.cuda.manual_seed(seed)


        torch.cuda.manual_seed_all(seed)


        torch.backends.cudnn.deterministic = True


        torch.backends.cudnn.benchmark = False


    print(f"setting seed : {seed}")


set_seed(93)


def to_device(obj,dev):


    if isinstance(obj, torch.Tensor):


        return obj.to(dev,non_blocking=True)


    elif isinstance(obj, HeteroData):


        for key in obj.keys():


            if isinstance(obj[key], torch.Tensor):


                obj[key] = obj[key].to(dev)


        return obj


    elif isinstance(obj, dict):


        return {k: to_device(v,dev) for k, v in obj.items()}


    return obj


def load_integrated_data(output_dir=None, device='cuda'):
    if output_dir is None:
        output_dir = get_data_dir()


    print("---------Loading integrated data...")


    hetero_graph = torch.load(os.path.join(output_dir, 'hetero_graph.pt'), weights_only=False)


    drug_disease_edges = torch.load(os.path.join(output_dir, 'drug_disease_edges.pt'), weights_only=False)


    node_features = torch.load(os.path.join(output_dir, 'node_features.pt'), weights_only=False)


    hetero_graph = to_device(hetero_graph,dev=device)


    drug_disease_edges = to_device(drug_disease_edges,dev=device)


    node_features = to_device(node_features,dev=device)


    logging.info(f"Data loaded from {output_dir} onto {device}")


    return hetero_graph, drug_disease_edges, node_features


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


print(f"Using device: {device}")


from torch_geometric.data import HeteroData


import pickle


import os


def generate_negative_samples(pos_edges, num_drugs, num_diseases, num_neg_samples=None):


    if num_neg_samples is None:


        num_neg_samples = pos_edges.shape[1]


    pos_set = set()


    for i in range(pos_edges.shape[1]):


        pos_set.add((pos_edges[0, i].item(), pos_edges[1, i].item()))


    neg_edges = []


    while len(neg_edges) < num_neg_samples:


        drug_idx = torch.randint(0, num_drugs, (1,)).item()


        disease_idx = torch.randint(0, num_diseases, (1,)).item()


        if (drug_idx, disease_idx) not in pos_set:


            neg_edges.append([drug_idx, disease_idx])


    neg_edges = torch.tensor(neg_edges, dtype=torch.long).t()


    return neg_edges


def create_dataset(hetero_data, drug_disease_edges, save_dir='./data'):


    os.makedirs(save_dir, exist_ok=True)


    num_drugs = hetero_data['drug'].x.shape[0]


    num_diseases = hetero_data['disease'].x.shape[0]


    print(f"Drug count: {num_drugs}")


    print(f"Disease count: {num_diseases}")


    print(f"Positive sample count: {drug_disease_edges.shape[1]}")


    neg_edges = generate_negative_samples(


        drug_disease_edges,


        num_drugs,


        num_diseases,


        num_neg_samples=drug_disease_edges.shape[1]


    )


    print(f"Negative sample count: {neg_edges.shape[1]}")


    all_data_pos_neg={


        'pos_edges': drug_disease_edges,


        'neg_edges': neg_edges,


        'labels': torch.cat([


            torch.ones(drug_disease_edges.shape[1]),


            torch.zeros(neg_edges.shape[1])


        ]).to(device)


    }


    torch.save(all_data_pos_neg, os.path.join(save_dir, 'all_data_pos_neg_pre_replacement.pt'))


    print(f"\nData saved to {save_dir} directory")


    return all_data_pos_neg


def load_dataset(save_dir='./data', device='cuda'):


    all_data_pos_neg = torch.load(os.path.join(save_dir, 'all_data_pos_neg_pre_replacement.pt'), weights_only=False, map_location=device)


    hetero_data = torch.load(os.path.join(save_dir, 'hetero_graph.pt'), weights_only=False, map_location=device)


    hetero_data = to_device(hetero_data,dev=device)


    all_data_pos_neg = to_device(all_data_pos_neg,dev=device)


    logging.info(f"Data loaded from {save_dir} onto {device}")


    return all_data_pos_neg, hetero_data


if __name__ == "__main__":


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(f"Using device: {device}, Available GPUs: {torch.cuda.device_count()}")


    torch.cuda.empty_cache()


    import gc


    gc.collect()


    hetero_graph, drug_disease_edges, node_features=load_integrated_data(output_dir=get_data_dir())


    save_path = get_output_dir()+"generated_pre_replacement/"


    os.makedirs(save_path, exist_ok=True)


    all_data_pos_neg=create_dataset(hetero_graph, drug_disease_edges,save_dir=save_path)

