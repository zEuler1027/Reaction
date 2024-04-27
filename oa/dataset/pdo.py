import torch
from torch.utils.data import Dataset
import numpy as np


class PDODataset(Dataset):
    def __init__(self, path: str, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.raw_data = np.load(path, allow_pickle=True)
        self.fragment = ['IS', 'TS', 'FS']
        self.n_fragment = 3
        self.data = []
        for reaction in range(len(self.raw_data['IS'])):
            data_item = {}  
            reaction_raw = [self.raw_data[frag][reaction] for frag in self.fragment]
            for ii in range(self.n_fragment):
                data_item[f'size_{ii}'] = torch.tensor(len(reaction_raw[ii])).to(self.device)
                
                data_item[f'pos_{ii}'] = torch.tensor(
                    reaction_raw[ii][:, 1: ], dtype=torch.float32
                ).to(self.device)
                
                data_item[f'one_hot_{ii}'] = torch.zeros(
                    (len(reaction_raw[ii]), 2), dtype=torch.long
                ).to(self.device)
                data_item[f'one_hot_{ii}'][ :135, 0] = torch.ones(135, dtype=torch.long)
                data_item[f'one_hot_{ii}'][135:, 1] = torch.ones(
                    len(reaction_raw[ii]) - 135, dtype=torch.long
                )
                
                data_item[f'charge_{ii}'] = torch.tensor(
                    reaction_raw[ii][:, 0], dtype=torch.long
                ).reshape(-1, 1).to(self.device)
                
                data_item[f'mask_{ii}'] = torch.zeros(
                    (len(reaction_raw[ii]), ), dtype=torch.long
                ).to(self.device)
                
                data_item['condition'] = torch.tensor(
                    [[0]], dtype=torch.long
                ).to(self.device)
                
            self.data.append(data_item)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]
    
    @staticmethod
    def collate_fn(batch):
        sizes = []
        for k in batch[0].keys():
            if "size" in k:
                sizes.append(int(k.split("_")[-1]))
        n_fragment = len(sizes)
        out = [{} for _ in range(n_fragment)]
        res = {}
        for prop in batch[0].keys():
            if prop not in ["condition", "target", "rmsd", "ediff"]:
                idx = int(prop.split("_")[-1])
                _prop = prop.replace(f"_{idx}", "")
            if "size" in prop:
                out[idx][_prop] = torch.tensor(
                    [x[prop] for x in batch],
                    device=batch[0][prop].device,
                )
            elif "mask" in prop:
                # make sure indices in batch start at zero (needed for
                # torch_scatter)
                out[idx][_prop] = torch.cat(
                    [
                        i * torch.ones(len(x[prop]), device=x[prop].device).long()
                        for i, x in enumerate(batch)
                    ],
                    dim=0,
                )
            elif prop in ["condition", "target", "rmsd", "ediff"]:
                res[prop] = torch.cat([x[prop] for x in batch], dim=0)
            else:
                out[idx][_prop] = torch.cat([x[prop] for x in batch], dim=0)
        if len(list(res.keys())) == 1:
            return out, res["condition"]
        return out, res
        

if __name__ == '__main__':
    data_path = 'oa/data/pdo/train.npz'
    dataset = PDODataset(data_path)
    print(len(dataset))
    print(dataset[0])
    
