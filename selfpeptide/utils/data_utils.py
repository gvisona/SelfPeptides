import numpy as np
import h5py
from tqdm import tqdm
import os
import torch
from torch.utils.data import Dataset, DataLoader
from selfpeptide.utils.constants import *

class PeptideTripletsDataset(Dataset):
    def __init__(self, hdf5_dataset_fname, gen_size=1000, init_random_state=None, hold_out_set=None):
        self.hdf5_dataset_fname = hdf5_dataset_fname
        self.gen_size =gen_size
        self.hold_out_set = hold_out_set
        
        if not os.path.exists(self.hdf5_dataset_fname):
            raise FileNotFoundError("Specify a valid HDF5 file for the dataset")
        self._get_n_peptides()
        self._generate_triplets(n_triplets=gen_size, random_state=init_random_state)
        
    def _get_n_peptides(self):
        peptides_n_ref = {}
        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            for k in f.keys():
                peptides_n_ref[k] = len(f[k])
        self.peptides_n_ref = peptides_n_ref
        
    def get_stored_peptides(self):
        peptides_set = set()
        for t in self.triplets:
            peptides_set.update(t)
        return peptides_set
                    
    def _generate_triplets(self, n_triplets=1000, random_state=None):
        triplets = []
        
        if random_state is not None:
            np.random.seed(random_state)
        
        pbar = tqdm(total=n_triplets)
        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            while len(triplets)<n_triplets:
                idx_p1 = np.random.randint(0, self.peptides_n_ref['reference_human_proteome'])
                idx_p2 = np.random.randint(0, self.peptides_n_ref['reference_human_proteome'])
                p1 = f['reference_human_proteome'][idx_p1].decode()
                p2 = f['reference_human_proteome'][idx_p2].decode()
                
                if p1==p2:
                    continue
                
                cat = np.random.choice(['bacterial_peptides', 'human_cancer_peptides', 'viral_peptides'])
                idx_neg = np.random.randint(0, self.peptides_n_ref[cat])
                neg = f[cat][idx_neg].decode()
                
                if self.hold_out_set is not None:
                    if (p1 in self.hold_out_set or 
                            p2 in self.hold_out_set or
                            neg in self.hold_out_set):
                        continue
                
                triplets.append([p1, p2, neg])
                pbar.update(1)
        pbar.close()
        self.triplets = triplets
        
        
    def refresh_data(self):
        self._generate_triplets(n_triplets=self.gen_size)
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]
    
    


class PeptideDataset_forMining(Dataset):
    def __init__(self, hdf5_dataset_fname, gen_size=1000, init_random_state=None, hold_out_set=None):
        self.hdf5_dataset_fname = hdf5_dataset_fname
        self.gen_size =gen_size
        self.hold_out_set = hold_out_set
        
        if not os.path.exists(self.hdf5_dataset_fname):
            raise FileNotFoundError("Specify a valid HDF5 file for the dataset")
        self._get_n_peptides()
        self._generate_peptides(n_peptides=gen_size, random_state=init_random_state)
        
    def _get_n_peptides(self):
        peptides_n_ref = {}
        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            for k in f.keys():
                peptides_n_ref[k] = len(f[k])
        self.peptides_n_ref = peptides_n_ref
        
    def get_stored_peptides(self):
        return set(self.peptides)
                    
    def _generate_peptides(self, n_peptides=10000, random_state=None):
        peptides = []
        labels = []
        
        if random_state is not None:
            np.random.seed(random_state)
        
        pbar = tqdm(total=n_peptides)
        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            while len(peptides)<n_peptides:
                idx_pos = np.random.randint(0, self.peptides_n_ref['reference_human_proteome'])
                pos = f['reference_human_proteome'][idx_pos].decode()

                cat = np.random.choice(['bacterial_peptides', 'human_cancer_peptides', 'viral_peptides'])
                idx_neg = np.random.randint(0, self.peptides_n_ref[cat])
                neg = f[cat][idx_neg].decode()
                
                if self.hold_out_set is not None:
                    if (pos in self.hold_out_set or
                        neg in self.hold_out_set):
                        continue
                
                peptides.append(pos)
                labels.append(1)
                peptides.append(neg)
                labels.append(0)
                pbar.update(2)
                
        pbar.close()
        self.peptides = peptides
        self.labels = labels
        
        
    def refresh_data(self):
        self._generate_peptides(n_peptides=self.gen_size)
    
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]
    
    
    



class Self_NonSelf_PeptideDataset(Dataset):
    def __init__(self, hdf5_dataset_fname, gen_size=1000, 
                 init_random_state=None, 
                 hold_out_set=None,
                 negative_label=-1):
        self.hdf5_dataset_fname = hdf5_dataset_fname
        self.gen_size = gen_size
        self.negative_label = negative_label
        self.self_hold_out_idx_set = None
        self.nonself_hold_out_idx_set = None
        
        self.idx_self_processed = set()
        self.idx_nonself_processed = set()
        
        if hold_out_set is not None:
            self.self_hold_out_idx_set, self.nonself_hold_out_idx_set = hold_out_set
            self.idx_self_processed.update(self.self_hold_out_idx_set)
            self.idx_nonself_processed.update(self.nonself_hold_out_idx_set)
        
        if not os.path.exists(self.hdf5_dataset_fname):
            raise FileNotFoundError("Specify a valid HDF5 file for the dataset")
        self._get_n_peptides()
        self._generate_peptides(n_peptides=gen_size, random_state=init_random_state)
        
    def _get_n_peptides(self):
        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            self.n_self_peptides = len(f["reference_human_peptides"])
            self.n_nonself_peptides = len(f["nonself_peptides"])
        
    def get_stored_peptides(self):
        return []
        # return set(self.peptides)
                    
    def _load_peptides(self, n_peptides=10000, random_state=None):
        peptides = torch.zeros((n_peptides, MAX_PEPTIDE_LEN)).long()
        labels = torch.ones(n_peptides).long()
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self_peptides_idxs = np.random.choice([i for i in range(self.n_self_peptides) 
                                               if i not in self.idx_self_processed], 
                                              size=n_peptides, replace=False)
        
        
        
        nonself_peptides_idxs = np.random.choice([i for i in range(self.n_nonself_peptides) 
                                               if i not in self.idx_nonself_processed], 
                                              size=n_peptides, replace=False) 
        
        
        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            peptides[::2, :] = f["reference_human_peptides"][self_peptides_idxs]
            peptides[1::2, :] = f["reference_human_peptides"][nonself_peptides_idxs]
            
        labels[1::2] = self.negative_label
        
        self.peptides = peptides
        self.labels = labels
        
        self.idx_self_processed.update(self_peptides_idxs)
        self.idx_nonself_processed.update(nonself_peptides_idxs)
        
        
    def refresh_data(self):
        if self.n_self_peptides-len(self.idx_self_processed)<self.gen_size:
            self.idx_self_processed = set()
            self.idx_nonself_processed = set()
        
            if self.self_hold_out_idx_set is not None:
                self.idx_self_processed.update(self.self_hold_out_idx_set)
                self.idx_nonself_processed.update(self.nonself_hold_out_idx_set)
        
        self._generate_peptides(n_peptides=self.gen_size)
    
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]