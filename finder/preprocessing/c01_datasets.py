

import os
from origin_runtime import get_project_dir
import random


import pandas as pd


import numpy as np


import re


import pickle


import sys


import time


project_dir = get_project_dir()


data_procesed_dir = project_dir+"data/processed/"


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


import json


import gzip


import os


import torch


import pandas as pd


import numpy as np


from torch_geometric.data import HeteroData, Data


from rdkit import Chem


from torch_geometric.nn import GCNConv


from transformers import AutoTokenizer, AutoModel


from torch.utils.data import DataLoader, TensorDataset


import logging


def set_seed(seed=42):


    random.seed(seed)


    np.random.seed(seed)


    torch.manual_seed(seed)


    os.environ['PYTHONHASHSEED'] = str(seed)


    if torch.cuda.is_available():


        torch.cuda.manual_seed(seed)


        torch.cuda.manual_seed_all(seed)


        torch.backends.cudnn.deterministic = True


        torch.backends.cudnn.benchmark = False


    print(f"setting seed : {seed}")


FILE_COLUMN_MAPPING = {


    'drug_disease': {


        'id1': 'drug_id', 'name1': 'drug_canonicalSmiles',


        'id2': 'disease_id', 'name2': 'disease_name'


    },


    'disease_target': {


        'id1': 'disease_id', 'name1': 'disease_name',


        'id2': 'target_id_new', 'name2': 'target_approvedSymbol'


    },


    'drug_target': {


        'id1': 'drug_id', 'name1': 'drug_canonicalSmiles',


        'id2': 'target_id_new', 'name2': 'target_approvedSymbol'


    },


    'target_function': {


        'id1': 'hgnc_id', 'name1': 'symbol',


        'id2': 'go_id', 'name2': 'lbl'


    },


    'function_function': {


        'id1': 'go_id_sub', 'name1': 'lbl_sub',


        'id2': 'go_id_obj', 'name2': 'lbl_obj'


    },


    'ppi': {


        'id1': 'hgnc_id_1', 'name1': 'symbol_1',


        'id2': 'hgnc_id_2', 'name2': 'symbol_2'


    },


    'gene_tf': {


        'id1': 'hgnc_id', 'name1': 'symbol',


        'id2': 'tf_id', 'name2': 'Name.TF'


    },


    'tf_mirna': {


        'id1': 'tf_id', 'name1': 'tf',


        'id2': 'mirna_id', 'name2': 'mir'


    },


    'mirna_gene': {


        'id1': 'mirna_id', 'name1': 'miRNA',


        'id2': 'hgnc_id', 'name2': 'symbol'


    },


    'drug_info': {


        'id': 'drug_id', 'name': 'drug_canonicalSmiles'


    },


    'disease_info': {


        'id': 'disease_id', 'name': 'disease_name'


    }


}


def smiles_to_molgraph_embedding(smiles_list,device, feature_dim=64):


    print("---------smiles_to_molgraph_embedding---------")


    class MolGraphGCN(torch.nn.Module):


        def __init__(self, in_channels, out_channels):


            super().__init__()


            self.conv1 = GCNConv(in_channels, 128)


            self.conv2 = GCNConv(128, out_channels)


        def forward(self, x, edge_index):


            x = self.conv1(x, edge_index).relu()


            x = self.conv2(x, edge_index)


            return x.mean(dim=0)


    embeddings = []


    gcn = MolGraphGCN(in_channels=3, out_channels=feature_dim).to(device)


    for smiles in smiles_list:


        mol = Chem.MolFromSmiles(smiles)


        if mol is None:


            logging.warning(f"Invalid SMILES: {smiles}")


            embeddings.append(torch.zeros(feature_dim, device=device))


            continue


        if mol.GetNumBonds() == 0:


            std_smiles = Chem.MolToSmiles(mol)


            mol = Chem.MolFromSmiles(std_smiles)


            print(f"new----Atoms: {mol.GetNumAtoms()}, Bonds: {mol.GetNumBonds()}")


            logging.warning(f"Invalid SMILES structure: {smiles}")


            continue


        try:


            Chem.SanitizeMol(mol)


        except:


            logging.warning(f"Invalid SMILES structure: {smiles}")


            embeddings.append(torch.zeros(feature_dim, device=device))


            continue


        atom_features = []


        for atom in mol.GetAtoms():


            atom_features.append([atom.GetAtomicNum(), atom.GetDegree(), atom.GetFormalCharge()])


        x = torch.tensor(atom_features, dtype=torch.float, device=device)


        edge_index = []


        for bond in mol.GetBonds():


            edge_index.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])


            edge_index.append([bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()])


        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t().contiguous()


        with torch.no_grad():


            embedding = gcn(x, edge_index)


        embeddings.append(embedding)


    return torch.stack(embeddings)


def get_biobert_embedding(text_list,device, feature_dim=64):


    print("---------get_biobert_embedding---------")


    try:


        model = AutoModel.from_pretrained(project_dir+"/models/biobert_base_cased_v12/").to(device)


        tokenizer = AutoTokenizer.from_pretrained(project_dir+"/models/biobert_base_cased_v12/")


        inputs = tokenizer(text_list, return_tensors="pt", padding=True, truncation=True, max_length=512)


        inputs = {k: v.to(device) for k, v in inputs.items()}


        with torch.no_grad():


            outputs = model(**inputs)


        embeddings = outputs.last_hidden_state.mean(dim=1)


        linear = torch.nn.Linear(768, feature_dim).to(device)


        return linear(embeddings)


    except Exception as e:


        logging.warning(f"BioBERT embedding failed: {e}")


        return torch.randn(len(text_list), feature_dim, device=device)


def load_data():


    data_files = {


            'disease_target': 'disease_target_direct_association',


            'drug_target': 'Drug_targets',


            'target_function': 'go_pathway_molcular_biofunction',


            'function_function': 'go_pathways_interaction',


            'ppi': 'string_ppi_links',


            'gene_tf': 'gene_tf',


            'tf_mirna': 'tf_mirna',


            'mirna_gene': 'mirna_target',


            'drug_disease': 'drug_approved_diseases',


            'drug_info': 'drug_basic_info_opentargets',


            'disease_info': 'disease_phenotype_new'


        }


    data = {}


    for key, file in data_files.items():


        try:


            df = pickle_load_obj(file, data_procesed_dir)


            if key in FILE_COLUMN_MAPPING:


                mapping = FILE_COLUMN_MAPPING[key]


                if 'id1' in mapping:


                    df = df.rename(columns={


                        mapping['id1']: 'id1',


                        mapping['name1']: 'name1',


                        mapping['id2']: 'id2',


                        mapping['name2']: 'name2'


                    })


                else:


                    df = df.rename(columns={


                        mapping['id']: 'id1',


                        mapping['name']: 'name1'


                    })


                if 'id1' not in df.columns:


                    df['id1'] = df['name1'].factorize()[0]


                if 'id2' in df.columns and 'id2' not in df:


                    df['id2'] = df['name2'].factorize()[0]


            df['weight'] = 1.0


            data[key] = df


        except Exception as e:


            logging.error(f"Failed to load {file}: {e}")


            data[key] = pd.DataFrame()


    return data


def build_subgraph_without_drug_disease(data, node_features,device):


    print("---------build_subgraph_without_drug_disease---------")


    sub_hetero_graph = HeteroData()


    sub_node_types = ['gene', 'function', 'tf', 'mirna']


    for node_type in sub_node_types:


        sub_hetero_graph[node_type].x = torch.randn(len(node_features[f'{node_type}_id_map']), 64, device=device)


    sub_edge_types = [


        ('gene', 'has_function', 'function'),


        ('function', 'interacts', 'function'),


        ('gene', 'regulated_by', 'tf'),


        ('tf', 'regulates', 'mirna'),


        ('mirna', 'regulates', 'gene'),


        ('gene', 'interacts', 'gene')


    ]


    key_map = {


        ('gene', 'has_function', 'function'): 'target_function',


        ('function', 'interacts', 'function'):"function_function",


        ('gene', 'regulated_by', 'tf'):"gene_tf",


        ('tf', 'regulates', 'mirna'):"tf_mirna",


        ('mirna', 'regulates', 'gene'):"mirna_gene",


        ('gene', 'interacts', 'gene'): 'ppi'


    }


    for src_type, relation, dst_type in sub_edge_types:


        if (src_type, relation, dst_type) == ('gene', 'has_function', 'function'):


            key = 'target_function'


        elif (src_type, relation, dst_type) == ('function', 'interacts', 'function'):


            key = 'function_function'


        elif (src_type, relation, dst_type) == ('gene', 'regulated_by', 'tf'):


            key = 'gene_tf'


        elif (src_type, relation, dst_type) == ('tf', 'regulates', 'mirna'):


            key = 'tf_mirna'


        elif (src_type, relation, dst_type) == ('mirna', 'regulates', 'gene'):


            key = 'mirna_gene'


        elif (src_type, relation, dst_type) == ('gene', 'interacts', 'gene'):


            key = 'ppi'


        else:


            continue


        if key in data and not data[key].empty:


            df = data[key]


            edges = [


                (node_features[f'{src_type}_id_map'][str(row['id1'])],


                 node_features[f'{dst_type}_id_map'][str(row['id2'])])


                for _, row in df.iterrows()


            ]


            edge_index = torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


            edge_weight = torch.ones(len(edges), device=device)


            sub_hetero_graph[src_type, relation, dst_type].edge_index = edge_index


            sub_hetero_graph[src_type, relation, dst_type].edge_weight = edge_weight


    return sub_hetero_graph


def get_subgraph_embedding(sub_hetero_graph, node_type, node_ids, node_features,device, feature_dim=64):


    print("---------get_subgraph_embedding---------")


    class SubGraphGCN(torch.nn.Module):


        def __init__(self, in_channels, out_channels):


            super().__init__()


            self.conv1 = GCNConv(in_channels, 128)


            self.conv2 = GCNConv(128, out_channels)


        def forward(self, x_dict, edge_index_dict):


            for nt in x_dict:


                for rel in ['interacts', 'has_function', 'regulated_by', 'regulates']:


                    edge_index = edge_index_dict.get((nt, rel, nt), torch.tensor([[], []], dtype=torch.long, device=device))


                    if edge_index.size(1) > 0:


                        x_dict[nt] = self.conv1(x_dict[nt], edge_index).relu()


                        x_dict[nt] = self.conv2(x_dict[nt], edge_index)


            return x_dict


    gcn = SubGraphGCN(in_channels=64, out_channels=feature_dim).to(device)


    with torch.no_grad():


        x_dict = {nt: sub_hetero_graph[nt].x for nt in sub_hetero_graph.node_types}


        edge_index_dict = {(src, rel, dst): sub_hetero_graph[src, rel, dst].edge_index


                           for src, rel, dst in sub_hetero_graph.edge_types}


        embeddings_dict = gcn(x_dict, edge_index_dict)


    embeddings = embeddings_dict[node_type]


    return embeddings


def initialize_node_features(data, node_types,device,feature_dim=64):


    node_features = {}


    try:


        drug_info = data['drug_info']


        disease_info = data['disease_info']


    except:


        drug_info = disease_info = None


        logging.warning("drug_info or disease_info not found")


    drug_ids_all = pd.concat([data['drug_disease']['id1'], data['drug_target']['id1']]).unique()


    valid_drug_ids = []


    drug_embeddings_list = []


    if drug_info is not None:


        for drug_id in drug_ids_all:


            smiles = drug_info[drug_info['id1'] == drug_id]['name1'].values


            if len(smiles) == 0:


                logging.warning(f"Drug {drug_id} has no SMILES; skipping")


                continue


            smiles = smiles[0]


            mol = Chem.MolFromSmiles(smiles)


            if mol is None or mol.GetNumBonds() == 0:


                logging.warning(f"Drug {drug_id} has invalid SMILES; skipping")


                continue


            embedding = smiles_to_molgraph_embedding([smiles], feature_dim)[0]


            drug_embeddings_list.append(embedding)


            valid_drug_ids.append(drug_id)


    node_features['drug_id_map'] = {str(id_): idx for idx, id_ in enumerate(valid_drug_ids)}


    node_features['drug'] = torch.stack(drug_embeddings_list) if valid_drug_ids else torch.tensor([], device=device)


    disease_ids_all = pd.concat([data['drug_disease']['id2'], data['disease_target']['id1']]).unique()


    valid_disease_ids = []


    disease_embeddings_list = []


    if disease_info is not None:


        for disease_id in disease_ids_all:


            disease_text = disease_info[disease_info['id1'] == disease_id]['name1'].values


            if len(disease_text) == 0:


                logging.warning(f"Disease {disease_id} has no text information; skipping")


                continue


            disease_text = disease_text[0]


            try:


                embedding = get_biobert_embedding([disease_text], feature_dim)[0]


                disease_embeddings_list.append(embedding)


                valid_disease_ids.append(disease_id)


            except Exception as e:


                logging.warning(f"Disease {disease_id} embedding failed: {e}; skipping")


    node_features['disease_id_map'] = {str(id_): idx for idx, id_ in enumerate(valid_disease_ids)}


    node_features['disease'] = torch.stack(disease_embeddings_list) if valid_disease_ids else torch.tensor([], device=device)


    for node_type in ['gene', 'function', 'tf', 'mirna']:


        if node_type == 'gene':


            node_ids = pd.concat([


                data['disease_target']['id2'], data['drug_target']['id2'],


                data['target_function']['id1'], data['ppi']['id1'], data['ppi']['id2'],


                data['gene_tf']['id1'], data['mirna_gene']['id2']


            ]).unique()


        elif node_type == 'function':


            node_ids = pd.concat([data['target_function']['id2'], data['function_function']['id1'], data['function_function']['id2']]).unique()


        elif node_type == 'tf':


            node_ids = pd.concat([data['gene_tf']['id2'], data['tf_mirna']['id1']]).unique()


        elif node_type == 'mirna':


            node_ids = pd.concat([data['tf_mirna']['id2'], data['mirna_gene']['id1']]).unique()


        node_features[f'{node_type}_id_map'] = {str(id_): idx for idx, id_ in enumerate(node_ids)}


    sub_hetero_graph = build_subgraph_without_drug_disease(data, node_features,device)


    for node_type in ['gene', 'function', 'tf', 'mirna']:


        node_ids = [k for k, _ in node_features[f'{node_type}_id_map'].items()]


        embeddings = get_subgraph_embedding(sub_hetero_graph, node_type, node_ids, node_features,device, feature_dim)


        node_features[node_type] = embeddings


    return node_features


def build_hetero_graph(data, node_features,device):


    print("---------Building hetero graph...")


    hetero_graph = HeteroData()


    node_types = ['gene', 'function', 'tf', 'mirna', 'drug', 'disease']


    for node_type in node_types:


        hetero_graph[node_type].x = node_features[node_type].to(device)


    edge_type_data_map = {


        ('gene', 'has_function', 'function'): "target_function",


        ('function', 'interacts', 'function'): "function_function",


        ('gene', 'regulated_by', 'tf'): "gene_tf",


        ('tf', 'regulates', 'mirna'): "tf_mirna",


        ('mirna', 'regulates', 'gene'): "mirna_gene",


        ('gene', 'interacts', 'gene'): "ppi",


        ('drug', 'targets', 'gene'): "drug_target",


        ('disease', 'associated_with', 'gene'): "disease_target"


    }


    for (src_type, rel, dst_type), data_key in edge_type_data_map.items():


        if data_key not in data or data[data_key].empty:


            continue


        df = data[data_key]


        src_id_map = node_features[f'{src_type}_id_map']


        dst_id_map = node_features[f'{dst_type}_id_map']


        valid_edges = []


        for _, row in df.iterrows():


            src_id_str = str(row['id1'])


            dst_id_str = str(row['id2'])


            if src_id_str in src_id_map and dst_id_str in dst_id_map:


                valid_edges.append((src_id_map[src_id_str], dst_id_map[dst_id_str]))


        if valid_edges:


            edge_index = torch.tensor(valid_edges, dtype=torch.long, device=device).t().contiguous()


            edge_weight = torch.ones(len(valid_edges), device=device)


            hetero_graph[src_type, rel, dst_type].edge_index = edge_index


            hetero_graph[src_type, rel, dst_type].edge_weight = edge_weight


        else:


            logging.warning(f"Edge type {src_type}-{rel}-{dst_type} has no valid edges; skipping")


    from sklearn.metrics.pairwise import cosine_similarity


    valid_drug_ids = list(node_features['drug_id_map'].keys())


    valid_disease_ids = list(node_features['disease_id_map'].keys())


    drug_emb_np = node_features['drug'].cpu().numpy()


    sim_matrix = cosine_similarity(drug_emb_np)


    threshold = 0.1


    similar_edges = []


    similar_edges_weight=[]


    for i in range(len(valid_drug_ids)):


        for j in range(i + 1, len(valid_drug_ids)):


            if sim_matrix[i, j] > threshold:


                similar_edges.append((valid_drug_ids[i], valid_drug_ids[j]))


                similar_edges.append((valid_drug_ids[j], valid_drug_ids[i]))


                similar_edges_weight.append(sim_matrix[i, j])


                similar_edges_weight.append(sim_matrix[i, j])


    src_indices = [node_features['drug_id_map'][s] for s, t in similar_edges]


    tgt_indices = [node_features['drug_id_map'][t] for s, t in similar_edges]


    hetero_graph['drug', 'similar', 'drug'].edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long).to(device)


    hetero_graph['drug', 'similar', 'drug'].edge_weight=torch.tensor(similar_edges_weight, dtype=torch.float32).to(device)


    print("drug similar_edges length:",len(similar_edges))


    disease_emb_np = node_features['disease'].detach().cpu().numpy()


    dis_sim_matrix = cosine_similarity(disease_emb_np)


    dis_similar_edges = []


    dis_similar_edges_weight=[]


    for i in range(len(valid_disease_ids)):


        for j in range(i + 1, len(valid_disease_ids)):


            if dis_sim_matrix[i, j] > threshold:


                dis_similar_edges.append((valid_disease_ids[i], valid_disease_ids[j]))


                dis_similar_edges.append((valid_disease_ids[j], valid_disease_ids[i]))


                dis_similar_edges_weight.append(dis_sim_matrix[i, j])


                dis_similar_edges_weight.append(dis_sim_matrix[i, j])


    src_indices = [node_features['disease_id_map'][s] for s, t in dis_similar_edges]


    tgt_indices = [node_features['disease_id_map'][t] for s, t in dis_similar_edges]


    hetero_graph['disease', 'similar', 'disease'].edge_index = torch.tensor([src_indices, tgt_indices], dtype=torch.long).to(device)


    hetero_graph['disease', 'similar', 'disease'].edge_weight=torch.tensor(dis_similar_edges_weight, dtype=torch.float32).to(device)


    print("disease similar_edges length:",len(dis_similar_edges))


    drug_disease_edges = []


    for _, row in data['drug_disease'].iterrows():


        drug_id_str = str(row['id1'])


        disease_id_str = str(row['id2'])


        if drug_id_str in node_features['drug_id_map'] and disease_id_str in node_features['disease_id_map']:


            drug_idx = node_features['drug_id_map'][drug_id_str]


            disease_idx = node_features['disease_id_map'][disease_id_str]


            drug_disease_edges.append((drug_idx, disease_idx))


    drug_disease_edges = torch.tensor(drug_disease_edges, dtype=torch.long, device=device).t() if drug_disease_edges else torch.tensor([], device=device)


    return hetero_graph, drug_disease_edges


def save_integrated_data(hetero_graph, drug_disease_edges, node_features, output_dir=project_dir+"models/code/"):


    print("---------Saving integrated data...")


    os.makedirs(output_dir, exist_ok=True)


    def to_cpu(obj):


        if isinstance(obj, torch.Tensor):


            return obj.cpu()


        elif isinstance(obj, HeteroData):


            for key in obj.keys():


                if isinstance(obj[key], torch.Tensor):


                    obj[key] = obj[key].cpu()


            return obj


        elif isinstance(obj, dict):


            return {k: to_cpu(v) for k, v in obj.items()}


        return obj


    hetero_graph_cpu = to_cpu(hetero_graph)


    torch.save(hetero_graph_cpu, os.path.join(output_dir, 'hetero_graph.pt'))


    drug_disease_edges_cpu = to_cpu(drug_disease_edges)


    torch.save(drug_disease_edges_cpu, os.path.join(output_dir, 'drug_disease_edges.pt'))


    node_features_cpu = to_cpu(node_features)


    torch.save(node_features_cpu, os.path.join(output_dir, 'node_features.pt'))


    metadata = {


        'node_types': hetero_graph.node_types,


        'edge_types': [(src, rel, dst) for src, rel, dst in hetero_graph.edge_types],


        'feature_dim': node_features['drug'].shape[1],


    }


    with open(os.path.join(output_dir, 'metadata.json'), 'w') as f:


        json.dump(metadata, f, indent=2)


    logging.info(f"Data saved to {output_dir}")


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


def load_integrated_data(output_dir=project_dir+"models/code/", device='cuda'):


    print("---------Loading integrated data...")


    hetero_graph = torch.load(os.path.join(output_dir, 'hetero_graph.pt'))


    drug_disease_edges = torch.load(os.path.join(output_dir, 'drug_disease_edges.pt'))


    node_features = torch.load(os.path.join(output_dir, 'node_features.pt'))


    hetero_graph = to_device(hetero_graph,dev=device)


    drug_disease_edges = to_device(drug_disease_edges,dev=device)


    node_features = to_device(node_features,dev=device)


    logging.info(f"Data loaded from {output_dir} onto {device}")


    return hetero_graph, drug_disease_edges, node_features


def main():


    set_seed(93)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    print(f"Using device: {device}")


    data = load_data()


    node_types = ['gene', 'function', 'tf', 'mirna', 'drug', 'disease']


    if os.path.exists(project_dir+"models/code/"+"node_features.pt"):


        node_features = torch.load(project_dir+"models/code/"+'node_features.pt')


        node_features = to_device(node_features,dev=device)


    else:


        node_features = initialize_node_features(data, node_types, feature_dim=64)


    hetero_graph, drug_disease_edges = build_hetero_graph(data, node_features,device)


    print("Heterogeneous graph structure:")


    print(hetero_graph)


    print("Drug-disease edge shape:", drug_disease_edges.shape)


    save_integrated_data(hetero_graph=hetero_graph, drug_disease_edges=drug_disease_edges, node_features=node_features, output_dir=project_dir+"models/code/data/all/")


    return hetero_graph, drug_disease_edges, node_features


if __name__ == "__main__":


    hetero_graph, drug_disease_edges, node_features = main()
