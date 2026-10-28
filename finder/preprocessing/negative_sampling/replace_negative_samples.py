"""Replace selected negative columns; preserve positives, labels and other columns."""
import argparse
import csv
from pathlib import Path
import torch


def draw_pairs(num_drugs, num_diseases, count, forbidden, seed):
    if num_drugs <= 0 or num_diseases <= 0 or count < 0:
        raise ValueError('Invalid sample domain or count')
    blocked=set(forbidden)
    in_domain=sum(0 <= d < num_drugs and 0 <= s < num_diseases for d,s in blocked)
    if count > num_drugs*num_diseases-in_domain:
        raise ValueError('Not enough available unique pairs')
    generator=torch.Generator(device='cpu').manual_seed(seed)
    pairs=[]
    for _ in range(1000000):
        if len(pairs)==count:
            return pairs
        flat=int(torch.randint(num_drugs*num_diseases,(1,),generator=generator))
        pair=divmod(flat,num_diseases)
        if pair not in blocked:
            pairs.append(pair);blocked.add(pair)
    raise ValueError('Sampling attempt limit reached')


def replace_pairs(data, replacements):
    pos=data['pos_edges'];neg=data['neg_edges'];labels=data['labels']
    if pos.ndim!=2 or neg.ndim!=2 or pos.shape[0]!=2 or neg.shape[0]!=2 or labels.numel()!=pos.shape[1]+neg.shape[1]:
        raise ValueError('Invalid dataset shapes')
    forbidden=set(map(tuple,pos.T.tolist())) | set(map(tuple,neg.T.tolist()))
    seen=set()
    for index,pair in replacements.items():
        if not isinstance(index,int) or index < 0 or index >= neg.shape[1]:
            raise ValueError('Invalid negative column index')
        if len(pair)!=2 or any(not isinstance(v,int) or v<0 for v in pair):
            raise ValueError('Invalid replacement node indices')
        pair=tuple(pair)
        if pair in forbidden or pair in seen:
            raise ValueError('Replacement overlaps an original or replacement pair')
        seen.add(pair)
    result=dict(data);result['neg_edges']=neg.clone()
    for index,pair in replacements.items():
        result['neg_edges'][:,index]=torch.tensor(pair,dtype=neg.dtype,device=neg.device)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-data',required=True,type=Path)
    parser.add_argument('--replacements-csv',required=True,type=Path)
    parser.add_argument('--output-data',required=True,type=Path)
    args=parser.parse_args()
    if args.output_data.exists():
        raise ValueError('Refuse to overwrite an existing output dataset')
    data=torch.load(args.input_data,weights_only=True,map_location='cpu')
    replacements={}
    with args.replacements_csv.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            index=int(row['negative_index'])
            if index in replacements:
                raise ValueError('Repeated negative column index')
            if data['neg_edges'][:,index].tolist()!=[int(row['old_drug_index']),int(row['old_disease_index'])]:
                raise ValueError('Original pair does not match the replacement ledger')
            replacements[index]=(int(row['new_drug_index']),int(row['new_disease_index']))
    result=replace_pairs(data,replacements)
    args.output_data.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,args.output_data)
    print('Replaced',len(replacements),'negative columns:',sorted(replacements))


if __name__=='__main__':
    main()
