import os

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from selfpeptide.utils.constants import *


#################################################
# SNS datasets
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
                 val_size=0,
                 negative_label=-1, test_run=False):
        self.hdf5_dataset_fname = hdf5_dataset_fname
        self.gen_size = gen_size
        self.val_size = val_size//2
        self.negative_label = negative_label        
        self.test_run = test_run
        
        
        if not os.path.exists(self.hdf5_dataset_fname):
            raise FileNotFoundError("Specify a valid HDF5 file for the dataset")
        self._get_n_peptides()
        
        self.idx_self = self.val_size
        self.idx_nonself = self.val_size
    
        self._load_peptides(gen_size)
        
    def _get_n_peptides(self):
        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            self.n_self_peptides = len(f["reference_human_peptides"])
            self.n_nonself_peptides = len(f["nonself_peptides"])

                    
    def _load_peptides(self, n_peptides):
        peptides = torch.zeros((n_peptides, MAX_PEPTIDE_LEN)).long()
        labels = torch.ones(n_peptides).long()
        

        with h5py.File(self.hdf5_dataset_fname, 'r') as f:
            peptides[::2, :] = torch.from_numpy(f["reference_human_peptides"][self.idx_self:self.idx_self+n_peptides//2])
            peptides[1::2, :] = torch.from_numpy(f["nonself_peptides"][self.idx_nonself:self.idx_nonself+n_peptides//2])
            
        labels[1::2] = self.negative_label
        
        self.peptides = peptides.long()
        self.labels = labels.long()
        
        self.idx_self += n_peptides//2
        self.idx_nonself += n_peptides//2
        
        
    def refresh_data(self):
        if self.n_self_peptides-self.idx_self<self.gen_size:
            self.idx_self = self.val_size
        if self.n_nonself_peptides-self.idx_nonself<self.gen_size:
            self.idx_nonself = self.val_size
            
        if self.test_run:
            self.idx_self = self.val_size
            self.idx_nonself = self.val_size
        else:
            self._load_peptides(self.gen_size)
    
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, idx):
        return self.peptides[idx], self.labels[idx]
    
    
    
    
    
    
    
def split_pretokenized_data(h5py_file, holdout_sizes=[], random_state=None, test_run=False):
    if test_run:
        print("LOADING TEST RUN DATA: REMOVE FLAG IF TRAINING ACTUAL MODEL")
        with h5py.File(h5py_file, "r") as f:
            print("Loading data")
            self_peptides = f['reference_human_peptides'][:100000]
            nonself_peptides = f['nonself_peptides'][:100000]    
    else:
        with h5py.File(h5py_file, "r") as f:
            print("Loading data")
            self_peptides = f['reference_human_peptides'][:]
            nonself_peptides = f['nonself_peptides'][:]    
    if isinstance(holdout_sizes, (int, float)):
        holdout_sizes = [holdout_sizes]
        
    print("Shuffling data")
    np.random.seed(random_state)
    np.random.shuffle(self_peptides)
    np.random.shuffle(nonself_peptides)
    
    print("Splitting data")
    peptides_sets = []
    for s in tqdm(holdout_sizes):
        holdout_pos, self_peptides = np.split(self_peptides, [s//2])
        holdout_neg, nonself_peptides = np.split(nonself_peptides, [s//2])      
        peptides_sets.append((holdout_pos, holdout_neg))

    peptides_sets.append((self_peptides, nonself_peptides))
    
    return peptides_sets
        
        
class PreSplit_Self_NonSelf_PeptideDataset(Dataset):
    def __init__(self, pos_class_data, neg_class_data,
                 negative_label=-1):
        super().__init__()
        self.pos_class_data = pos_class_data
        self.neg_class_data = neg_class_data
        self.negative_label = negative_label        
        
        self.n_negative_samples = len(neg_class_data)
        self.n_positive_samples = len(pos_class_data)
        self.labels = [negative_label, 1]

    def __len__(self):
        return len(self.pos_class_data) + len(self.neg_class_data)
    
    def __getitem__(self, idx):
        mod_idx = idx//2
        sample_class = idx % 2
        if sample_class==0:
            mod_idx = mod_idx % self.n_negative_samples
            peptide = self.neg_class_data[mod_idx]
        elif sample_class==1:
            mod_idx = mod_idx % self.n_positive_samples
            peptide = self.pos_class_data[mod_idx]
        return peptide, self.labels[sample_class]
    
    
class PreTokenized_HumanPeptidesDataset(Dataset):
    def __init__(self, hdf5_dataset_fname, test_run=False):
        super().__init__()
        self.hdf5_dataset_fname = hdf5_dataset_fname
        if not os.path.exists(self.hdf5_dataset_fname):
            raise FileNotFoundError("Specify a valid HDF5 file for the dataset")
        
        
        if test_run:
            with h5py.File(self.hdf5_dataset_fname, 'r') as f:
                self.peptides = torch.from_numpy(f["reference_human_peptides"][:10000])
        else:
            with h5py.File(self.hdf5_dataset_fname, 'r') as f:
                self.peptides = torch.from_numpy(f["reference_human_peptides"][:])
        
    def __len__(self):
        return len(self.peptides)
    
    def __getitem__(self, idx):
        return self.peptides[idx]
    
    
    
    
#########################################################
# Binding affinity datasets

def filter_peptide_dataset(df, vocabulary, label="Peptide"):
    aa_vocab = set(vocabulary)
    
    ix_to_keep = []
    
    for i, t in enumerate(df.itertuples()):
        if len(set(getattr(t, label)).difference(aa_vocab))==0:
            ix_to_keep.append(i)
    return df.iloc[ix_to_keep]
    
    

    
def load_binding_affinity_dataframes(config, split_data=True):
    ps_df = pd.read_csv(config['pseudo_seq_file'], sep="\t")
    hla_mapping = dict(ps_df[["HLA", "sequence"]].values)

    ba_df = pd.read_csv(config['binding_affinity_df'])
    ba_df["Allele Pseudo-sequence"] = ba_df["HLA"].str.replace("*", "", regex=False).map(hla_mapping)
    ba_df = ba_df.dropna().reset_index(drop=True)
    
    ba_df = filter_peptide_dataset(ba_df, sorted_vocabulary)
    
    ba_df["Stratification_index"] = ba_df["HLA"] + "_" + ba_df["Label"].astype(str)
    if not split_data:
        return ba_df
    
    ix = ba_df["Stratification_index"].value_counts()
    low_count_labels = ix[ix<3].index
    res_df = ba_df[ba_df["Stratification_index"].isin(low_count_labels)]
    ba_df = ba_df[~ba_df["Stratification_index"].isin(low_count_labels)]

    
    trainval_ba_df, test_ba_df = train_test_split(ba_df, test_size=config["test_size"], stratify=ba_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    train_ba_df, val_ba_df = train_test_split(trainval_ba_df, test_size=config["val_size"], stratify=trainval_ba_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    if res_df is not None:
        train_ba_df = pd.concat([train_ba_df, res_df])
        
    return train_ba_df, val_ba_df, test_ba_df

class SequencesInteractionDataset(Dataset):
    def __init__(self, df, hla_repr="Allele Pseudo-sequence", target_label="Label"):        
        super().__init__()
        cols = ["Peptide", hla_repr, target_label]
        self.data_matrix = df[cols].values.tolist()
    
    def __len__(self):
        return len(self.data_matrix)
    
    def __getitem__(self, ix):
        return self.data_matrix[ix]