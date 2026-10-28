

import os
import copy
from origin_runtime import get_data_dir, get_output_dir


import pandas as pd


import numpy as np


import re


import pickle


import sys


import time


import json


import gzip


import logging


import random


import torch


import torch.nn as nn


import torch.nn.functional as F


from torch_geometric.data import HeteroData


import numpy as np


from typing import Dict, List, Tuple, Optional


import matplotlib.pyplot as plt


import seaborn as sns


from collections import defaultdict


from torch.utils.data import DataLoader, TensorDataset


from sklearn.model_selection import KFold


from sklearn.model_selection import StratifiedKFold


from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score


import gc


from tqdm import tqdm


import math


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


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


from torch_geometric.data import HeteroData


def to_device(obj,dev):


    if isinstance(obj, torch.Tensor):


        return obj.to(dev,non_blocking=True)


    elif isinstance(obj, HeteroData):


        return obj.to(dev)


    elif isinstance(obj, dict):


        return {k: to_device(v,dev) for k, v in obj.items()}


    return obj


def load_dataset(save_dir='./data', device='cuda'):


    all_data_pos_neg = torch.load(os.path.join(save_dir, 'all_data_pos_neg_post_replacement_8.pt'), weights_only=False, map_location=device)


    hetero_data = torch.load(os.path.join(save_dir, 'hetero_graph.pt'), weights_only=False, map_location=device)


    hetero_data = to_device(hetero_data,dev=device)


    all_data_pos_neg = to_device(all_data_pos_neg,dev=device)


    logging.info(f"Data loaded from {save_dir} onto {device}")


    return all_data_pos_neg, hetero_data


from torch_geometric.nn import HeteroConv, RGCNConv


class RGCNEncoder(nn.Module):


    def __init__(self, in_dim, hidden_dim, out_dim, etypes, num_bases=10, dev="cuda:0"):


        super().__init__()


        self.device = dev


        self.conv1 = HeteroConv({


            etype: RGCNConv(in_dim, hidden_dim, num_relations=len(etypes), num_bases=num_bases)


            for etype in etypes


        }, aggr='sum')


        self.conv2 = HeteroConv({


            etype: RGCNConv(hidden_dim, out_dim, num_relations=len(etypes), num_bases=num_bases)


            for etype in etypes


        }, aggr='sum')


        self.to(dev)


    def forward(self, data):


        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict


        edge_type_dict = {edge_type: data[edge_type].edge_type for edge_type in data.edge_types}


        x_dict = self.conv1(x_dict, edge_index_dict,edge_type_dict)


        x_dict = {k: F.relu(v) for k, v in x_dict.items()}


        x_dict = self.conv2(x_dict, edge_index_dict,edge_type_dict)


        x_dict = {k: F.relu(v) for k, v in x_dict.items()}


        return x_dict


class FunctionAttentionModule(nn.Module):


    def __init__(self, embedding_dim, num_heads=4, dev="cuda:0"):


        super().__init__()


        self.device = dev


        self.attention = nn.MultiheadAttention(


            embed_dim=embedding_dim,


            num_heads=num_heads,


            batch_first=True


        )


        self.attention=self.attention.to(self.device)


    def forward(self,x_dict,batch_idx,entity_type: str = 'drug'):


        query = x_dict[entity_type][batch_idx].unsqueeze(1)


        h_functions = x_dict['function'].unsqueeze(0).repeat(len(batch_idx), 1, 1)


        output, attn_weights = self.attention(query, h_functions, h_functions)


        v_res = attn_weights.mean(dim=1)


        return v_res


class PredictionModule(nn.Module):


    def __init__(self, num_functions,hidden_dim=256,dropout=0.3, device="cuda:0"):


        super().__init__()


        self.device = device


        input_dim = num_functions * 4 + 1


        self.predictor = nn.Sequential(


            nn.Linear(input_dim, hidden_dim),


            nn.ReLU(),


            nn.LayerNorm(hidden_dim),


            nn.Dropout(dropout),


            nn.Linear(hidden_dim, hidden_dim // 2),


            nn.ReLU(),


            nn.LayerNorm(hidden_dim // 2),


            nn.Linear(hidden_dim // 2, 1)


        )


        self.to(device)


    def forward(self, drug_func_reps, disease_func_reps):


        sim = F.cosine_similarity(drug_func_reps, disease_func_reps, dim=-1).unsqueeze(-1)


        hadamard = drug_func_reps * disease_func_reps


        abs_diff = torch.abs(drug_func_reps - disease_func_reps)


        concat_features = torch.cat([


            drug_func_reps, disease_func_reps, hadamard ,abs_diff, sim


        ], dim=-1)


        result=self.predictor(concat_features)


        return result


class FuncRepNet(nn.Module):


    def __init__(self, original_dim, projected_dim, num_functions, etypes, num_heads=4, device=None):


        super().__init__()


        self.projected_dim = projected_dim
        self.device = torch.device(device if device is not None else
                                   ('cuda:0' if torch.cuda.is_available() else 'cpu'))


        self.graph_encoder = RGCNEncoder(


            in_dim=original_dim,


            hidden_dim=projected_dim,


            out_dim=projected_dim,


            etypes=etypes,


            dev=self.device


        )


        self.function_attention = FunctionAttentionModule(embedding_dim=projected_dim, num_heads=num_heads, dev=self.device).to(self.device)


        self.predictor = PredictionModule(num_functions=num_functions, device=self.device).to(self.device)


    def forward(self,data, drug_idxs, disease_idxs):


        x_dict = self.graph_encoder(data)


        drug_func_reps = self.function_attention(x_dict=x_dict,batch_idx=drug_idxs,entity_type='drug')


        print("drug_func_reps",drug_func_reps.size())


        disease_func_reps = self.function_attention(x_dict=x_dict,batch_idx=disease_idxs, entity_type='disease')


        print("disease_func_reps",disease_func_reps.size())


        logits = self.predictor(drug_func_reps.to(self.device), disease_func_reps.to(self.device))


        print("logits",logits.size())


        return logits.squeeze(-1).to(self.device)


def check_all_gpu_memory():


    if torch.cuda.is_available():


        gpu_count = torch.cuda.device_count()


        print(f"Detected {gpu_count} GPUs")


        for i in range(gpu_count):


            print(f"\n=== GPU {i} ===")


            allocated = torch.cuda.memory_allocated(i) / 1024**3


            cached = torch.cuda.memory_reserved(i) / 1024**3


            total = torch.cuda.get_device_properties(i).total_memory / 1024**3


            print(f"Device name: {torch.cuda.get_device_properties(i).name}")


            print(f"Total memory: {total:.2f} GB")


            print(f"Allocated: {allocated:.2f} GB ({allocated/total*100:.1f}%)")


            print(f"Cached: {cached:.2f} GB ({cached/total*100:.1f}%)")


            print(f"Available: {total-cached:.2f} GB")


    else:


        print("No GPU detected")


def cleanup_memory():


    import gc


    gc.collect()


    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


import torch


import gc


def full_clear_cache():


    import gc


    gc.collect()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def train_epoch(model, train_loader, optimizer, criterion, device, hetero_data):


    print("----------------------------Training----------------------")


    model.train()


    total_loss = 0.0


    all_preds = []


    all_labels = []


    for batch_drug_indices, batch_disease_indices, batch_labels in train_loader:


        batch_drug_indices = batch_drug_indices.to(device, non_blocking=True)


        batch_disease_indices = batch_disease_indices.to(device, non_blocking=True)


        batch_labels = batch_labels.to(device, non_blocking=True)


        optimizer.zero_grad(set_to_none=True)


        predictions = model(data=hetero_data,drug_idxs=batch_drug_indices, disease_idxs=batch_disease_indices)


        loss = criterion(predictions, batch_labels)


        loss.backward()


        optimizer.step()


        total_loss += loss.item()


        all_preds.extend(predictions.detach().cpu().numpy())


        all_labels.extend(batch_labels.detach().cpu().numpy())


        del predictions, loss, batch_drug_indices, batch_disease_indices, batch_labels


        if torch.cuda.is_available():
            torch.cuda.synchronize()


    return (total_loss / len(train_loader),np.array(all_preds, dtype=np.float32),np.array(all_labels, dtype=np.float32))


from sklearn.metrics import (


    roc_auc_score, average_precision_score, accuracy_score,


    precision_score, recall_score, f1_score, confusion_matrix


)


def evaluate_model(model, val_loader, criterion, device, hetero_data):


    model.eval()


    total_loss = 0.0
    total_samples = 0


    all_preds = []


    all_labels = []


    with torch.no_grad():


        for batch_drug_indices, batch_disease_indices, batch_labels in tqdm(val_loader, desc="Evaluating"):


            batch_drug_indices = batch_drug_indices.to(device)


            batch_disease_indices = batch_disease_indices.to(device)


            batch_labels = batch_labels.to(device)


            predictions = model(data=hetero_data,drug_idxs=batch_drug_indices, disease_idxs=batch_disease_indices)


            loss = criterion(predictions, batch_labels)


            batch_samples = batch_labels.size(0)
            total_loss += loss.item() * batch_samples
            total_samples += batch_samples


            all_preds.extend(predictions.detach().cpu().numpy())


            all_labels.extend(batch_labels.detach().cpu().numpy())


            del predictions, loss, batch_drug_indices, batch_disease_indices, batch_labels


    preds = np.array(all_preds)


    labels = np.array(all_labels)


    model_res={"preds":all_preds,"labels":all_labels}


    auc = roc_auc_score(labels, preds)


    aupr = average_precision_score(labels, preds)


    acc = accuracy_score(labels, preds > 0.5)


    precision = precision_score(labels, preds > 0.5, zero_division=0)


    recall = recall_score(labels, preds > 0.5, zero_division=0)


    f1 = f1_score(labels, preds > 0.5, zero_division=0)


    tn, fp, fn, tp = confusion_matrix(labels, preds > 0.5).ravel()


    specificity = tn / (tn + fp + 1e-8)


    # The original tuple retains logit > 0.5; also report probability > 0.5.
    probability_labels = preds > 0.0
    prob_tn, prob_fp, prob_fn, prob_tp = confusion_matrix(labels, probability_labels).ravel()
    model_res["probability_0_5_metrics"] = {
        "acc": accuracy_score(labels, probability_labels),
        "precision": precision_score(labels, probability_labels, zero_division=0),
        "recall": recall_score(labels, probability_labels, zero_division=0),
        "f1": f1_score(labels, probability_labels, zero_division=0),
        "specificity": prob_tn / (prob_tn + prob_fp + 1e-8),
    }
    model_res["logit_thresholds"] = {"original": 0.5, "probability_0_5": 0.0}

    avg_loss = total_loss / total_samples


    return avg_loss, auc, aupr, acc,precision,recall,f1,specificity,model_res


def k_fold_cross_validation(hetero_data, train_edges, train_labels,output_dir,


                            k=10, batch_size=32, num_epochs=50, lr=0.001, device='cuda'):


    fold_start = int(os.environ.get("FINDER_FOLD_START", "1"))
    fold_end = int(os.environ.get("FINDER_FOLD_END", "10"))
    if not 1 <= fold_start <= fold_end <= k:
        raise ValueError(f"FINDER fold bounds must satisfy 1 <= start <= end <= {k}")

    (f"Starting {k}-fold cross-validation...")


    edges_np = train_edges.cpu().numpy().T


    labels_np = train_labels.cpu().numpy()


    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


    fold_results = []


    fold_preds_labels=[]


    for fold, (train_idx, val_idx) in enumerate(skf.split(edges_np, labels_np)):
        # Skip before model creation or loader iteration consumes training RNG.
        if not fold_start <= fold + 1 <= fold_end:
            continue


        print(f"\n===== Fold {fold + 1}/{k} =====")


        train_edges_fold = torch.tensor(edges_np[train_idx].T, device=device)


        val_edges_fold = torch.tensor(edges_np[val_idx].T, device=device)


        train_labels_fold = torch.tensor(labels_np[train_idx], device=device)


        val_labels_fold = torch.tensor(labels_np[val_idx], device=device)


        train_dataset = TensorDataset(


            train_edges_fold[0], train_edges_fold[1], train_labels_fold


        )


        val_dataset = TensorDataset(


            val_edges_fold[0], val_edges_fold[1], val_labels_fold


        )


        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=False, drop_last=True,num_workers=0)


        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=False, drop_last=False,num_workers=0)


        model = FuncRepNet(


            original_dim=64,


            projected_dim=64,


            num_functions=hetero_data['function'].x.shape[0],


            num_heads=4,


            etypes=list(hetero_data.edge_types),
            device=device


        )


        model=model.to(device)


        print("Current model hyperparameter: ","original_dim=64,projected_dim=64,num_heads=4")


        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)


        criterion = nn.BCEWithLogitsLoss()


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)


        best_val_auc = float("-inf")


        patience_counter = 0


        max_patience = 20


        epoch_res_list=[]


        save_dir = output_dir+"fold_"+str(fold)+"/"


        if not os.path.exists(save_dir):


            os.makedirs(save_dir, exist_ok=True)


        for epoch in range(num_epochs):


            print("epoch:",epoch)


            train_loss, train_preds, train_labels_epoch = train_epoch(model, train_loader, optimizer, criterion, device, hetero_data=hetero_data)


            val_loss, val_auc, val_aupr, val_acc,val_precision,val_recall,val_f1,val_specificity,val_model_res= evaluate_model(model, val_loader, criterion, device, hetero_data=hetero_data)


            scheduler.step(val_loss)


            if val_auc > best_val_auc:


                best_val_auc = val_auc


                patience_counter = 0


                best_model_state = copy.deepcopy(model.state_dict())


            else:


                patience_counter += 1


            print(f"Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, "f"Val AUC={val_auc:.4f}, Val AUPR={val_aupr:.4f}, Val ACC (original logit > 0.5)={val_acc:.4f}, Val Precision={val_precision:.4f}, Val Recall={val_recall:.4f}, Val F1={val_f1:.4f}, Val Specificity={val_specificity:.4f}")
            print("Probability 0.5 (logit > 0.0): " + ", ".join(
                f"{name.upper()}={value:.4f}" for name, value in val_model_res["probability_0_5_metrics"].items()))


            epoch_res_list.append({'fold': fold + 1,"epoch":epoch,"train_loss":train_loss,"val_loss":val_loss, "val_auc":val_auc, "val_aupr":val_aupr, "val_acc":val_acc,"val_precision":val_precision,"val_recall":val_recall,"val_f1":val_f1,"val_specificity":val_specificity, **{"val_" + name + "_prob_0_5": value for name, value in val_model_res["probability_0_5_metrics"].items()}})


            if patience_counter > max_patience:


                print(f"Early stopping at epoch {epoch}")


                break


            check_all_gpu_memory()


            for gpu_id in range(torch.cuda.device_count()):


                with torch.cuda.device(gpu_id):


                    full_clear_cache()


            check_all_gpu_memory()


        # Keep the last completed epoch before restoring the best weights.
        last_model_state = copy.deepcopy(model.state_dict())


        epoch_res=pd.DataFrame(epoch_res_list)


        epoch_res.to_csv(os.path.join(save_dir, "flod_"+str(fold)+"_epoch_res.csv"))


        model.load_state_dict(best_model_state)


        final_val_loss, final_val_auc, final_val_aupr, final_val_acc,final_val_precision,final_val_recall,final_val_f1,final_val_specificity,final_val_model_res = evaluate_model(


            model, val_loader, criterion, device, hetero_data=hetero_data


        )


        fold_preds_labels.append({


            "fold":fold+1,


            "model_res":final_val_model_res


        })


        fold_results.append({


            'fold': fold + 1,


            'val_loss': final_val_loss,


            'val_auc': final_val_auc,


            'val_aupr': final_val_aupr,


            'val_acc': final_val_acc,


            "val_precision":final_val_precision,


            "val_recall":final_val_recall,


            "val_f1":final_val_f1,


            "val_specificity":final_val_specificity,
            **{"val_" + name + "_prob_0_5": value for name, value in final_val_model_res["probability_0_5_metrics"].items()}


        })


        print(f"Fold {fold + 1} Results: AUC={final_val_auc:.4f}, AUPR={final_val_aupr:.4f}, ACC (original logit > 0.5)={final_val_acc:.4f}, Precision={final_val_precision:.4f}, Recall={final_val_recall:.4f}, F1={final_val_f1:.4f}, Specificity={final_val_specificity:.4f}")
        print("Probability 0.5 (logit > 0.0): " + ", ".join(
            f"{name.upper()}={value:.4f}" for name, value in final_val_model_res["probability_0_5_metrics"].items()))


        torch.save(best_model_state, os.path.join(save_dir, "flod_"+str(fold)+"_bset_model_state_dict.pt"))
        torch.save(last_model_state, os.path.join(save_dir, "flod_"+str(fold)+"_last_model_state_dict.pt"))


        print(f"Training results saved to: {save_dir}")


        del model, optimizer, scheduler, train_loader, val_loader


        del train_dataset, val_dataset, best_model_state, last_model_state


        gc.collect()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


    with open(output_dir+"fold_preds_labels" + '.pkl', 'wb') as f:


        pickle.dump(fold_preds_labels, f, pickle.HIGHEST_PROTOCOL)


    avg_auc = np.mean([r['val_auc'] for r in fold_results])


    std_auc = np.std([r['val_auc'] for r in fold_results])


    avg_ap = np.mean([r['val_aupr'] for r in fold_results])


    std_ap = np.std([r['val_aupr'] for r in fold_results])


    avg_acc = np.mean([r['val_acc'] for r in fold_results])


    std_acc = np.std([r['val_acc'] for r in fold_results])


    print(f"\n===== Cross-validation results =====")


    print(f"AUC: {avg_auc:.4f} ± {std_auc:.4f}")


    print(f"AP:  {avg_ap:.4f} ± {std_ap:.4f}")


    for suffix, threshold_name in (("", "Original (logit > 0.5)"),
                                   ("_prob_0_5", "Probability 0.5 (logit > 0.0)")):
        print(threshold_name)
        for name in ("acc", "precision", "recall", "f1", "specificity"):
            values = [row["val_" + name + suffix] for row in fold_results]
            print(f"{name.upper()}: {np.mean(values):.4f} ± {np.std(values):.4f}")


    return fold_results, avg_auc, std_auc, avg_ap, std_ap, avg_acc, std_acc


def auto_add_edge_type(hetero_data: HeteroData) -> HeteroData:


    edge_types = hetero_data.edge_types


    for idx, edge_type in enumerate(edge_types):


        if 'edge_index' not in hetero_data[edge_type]:


            print(f"Warning: Edge type {edge_type} has no edge_index; skipping")


            continue


        num_edges = hetero_data[edge_type].edge_index.size(1)


        edge_type_tensor = torch.full((num_edges,), idx, dtype=torch.long)


        hetero_data[edge_type].edge_type = edge_type_tensor


        print(f"Added edge_type for {edge_type}, index={idx}, edge count={num_edges}")


    return hetero_data


if __name__ == "__main__":
    if os.environ.get("FINDER_RESUME", "never") != "never":
        raise SystemExit("FINDER Table 2 does not support checkpoint resume; use FINDER_RESUME=never")


    output_dir = get_output_dir()


    set_seed(93)


    device = torch.device(os.environ.get('FINDER_DEVICE', 'cuda:0' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu'))


    print(f"Using device: {device}")


    print(f"PyTorch Version: {torch.__version__}")


    print(f"CUDA Available: {torch.cuda.is_available()}")


    if torch.cuda.is_available():


        print(f"CUDA Version: {torch.version.cuda}")


        print(f"Current Device: {torch.cuda.get_device_name(0)}")


    save_path = get_data_dir()


    train_data, hetero_data = load_dataset(save_dir=save_path, device=device)


    hetero_data = auto_add_edge_type(hetero_data)


    hetero_data=hetero_data.to(device)


    train_data=to_device(train_data,dev=device)


    print(hetero_data)


    print("node_types:",hetero_data.node_types)


    print("edge_types:",hetero_data.edge_types)


    train_edges = torch.cat([train_data['pos_edges'], train_data['neg_edges']], dim=1)


    train_labels = train_data['labels']


    train_edges = train_edges.to(device)


    train_labels = train_labels.to(device)


    print("train_edges",train_edges.shape,train_edges)


    print("train_labels",train_labels.shape,train_labels)


    fold_results, avg_auc, std_auc, avg_ap, std_ap, avg_acc, std_acc = k_fold_cross_validation(


        hetero_data=hetero_data,


        train_edges=train_edges,


        train_labels=train_labels,


        output_dir=output_dir,


        k=10,


        batch_size=512,


        num_epochs=50,


        lr=0.001,


        device=device


    )


    print(fold_results)


    val_flod_df = pd.DataFrame(fold_results)


    val_flod_df.to_csv(output_dir+'cv_metrics_val.csv', index=False)


    print("Metrics saved to 'cv_metrics_val.csv'")


    gc.collect()


    if torch.cuda.is_available():
        torch.cuda.empty_cache()
