import os

import h5py
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from selfpeptide.utils.constants import *
from selfpeptide.utils.beta_distr_utils import *

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



def load_binding_affinity_dataframes_jointseqs(config, split_data=True):
    ps_df = pd.read_csv(config['pseudo_seq_file'])
    prot_df = pd.read_csv(config['hla_prot_seq_file'])


    hla_pseq_mapping = dict(ps_df[["HLA", "sequence"]].values)
    hla_prot_mapping = dict(prot_df[["HLA", "sequence"]].values)
    
    ba_df = pd.read_csv(config['binding_affinity_df'])
    ba_df["Allele Pseudo-sequence"] = ba_df["HLA"].str.replace("*", "", regex=False).map(hla_pseq_mapping)
    ba_df["Allele Protein sequence"] = ba_df["HLA"].str.replace("*", "", regex=False).map(hla_prot_mapping)
    ba_df = ba_df.dropna().reset_index(drop=True)
    ba_df = filter_peptide_dataset(ba_df, sorted_vocabulary)
    ba_df = ba_df[["HLA", "Peptide", "Label", "Allele Pseudo-sequence", "Allele Protein sequence"]]
    
    ligand_atlas_binding_df = pd.read_csv(config["ligand_atlas_binding_df"])
    ligand_atlas_binding_df["Allele Pseudo-sequence"] = ligand_atlas_binding_df["HLA"].str.replace("*", "", regex=False).map(hla_pseq_mapping)
    ligand_atlas_binding_df["Allele Protein sequence"] = ligand_atlas_binding_df["HLA"].str.replace("*", "", regex=False).map(hla_prot_mapping)
    ligand_atlas_binding_df = ligand_atlas_binding_df.dropna().reset_index(drop=True)
    ligand_atlas_binding_df = filter_peptide_dataset(ligand_atlas_binding_df, sorted_vocabulary)

    # Filter to remove duplicate samples 
    dhlap_samples = set(tuple(x) for x in ba_df[["Peptide", "HLA"]].values)
    la_samples = set(tuple(x) for x in ligand_atlas_binding_df[["Peptide", "HLA"]].values)
    ligand_atlas_binding_df = ligand_atlas_binding_df[ligand_atlas_binding_df[["Peptide", "HLA"]].apply(tuple, 1).isin(la_samples.difference(dhlap_samples))]
    
    ba_df = pd.concat([ba_df, ligand_atlas_binding_df])
    
    # ba_df
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
    def __init__(self, df, hla_repr=["Allele Pseudo-sequence"], target_label="Label"):        
        super().__init__()
        cols = ["Peptide", *hla_repr, target_label]
        self.data_matrix = df[cols].values.tolist()
    
    def __len__(self):
        return len(self.data_matrix)
    
    def __getitem__(self, ix):
        return self.data_matrix[ix]
    
    
class SequencesInteractionDataset_returnHLA(Dataset):
    def __init__(self, df, hla_repr="Allele Pseudo-sequence", target_label="Label"):        
        super().__init__()
        cols = ["Peptide", "HLA", hla_repr, target_label]
        self.data_matrix = df[cols].values.tolist()
    
    def __len__(self):
        return len(self.data_matrix)
    
    def __getitem__(self, ix):
        return self.data_matrix[ix]
    
#########################################################
# Immunogenicity datasets

   
def load_immunogenicity_dataframes(config, split_data=True):
    ps_df = pd.read_csv(config['pseudo_seq_file'], sep="\t")
    hla_mapping = dict(ps_df[["HLA", "sequence"]].values)

    res_df = None
    if config.get("trainval_df", None) is not None and config.get("test_df", None) is not None:
        print("Loading pre-computed training and test sets..")
        trainval_df = pd.read_csv(config["trainval_df"])
        test_df = pd.read_csv(config["test_df"])
    else:
        imm_beta_df = pd.read_csv(config['immunogenicity_df'])
        imm_beta_df["Allele Pseudo-sequence"] = imm_beta_df["HLA"].str.replace("*", "", regex=False).map(hla_mapping)
        imm_beta_df = imm_beta_df.dropna(subset=["Peptide", "Allele Pseudo-sequence"])
        imm_beta_df["Target"] = (imm_beta_df["Qualitative Measurement"]!="Negative").astype(int).values
        imm_beta_df["Sample"] = imm_beta_df["Peptide"] + "_" + imm_beta_df["HLA"]
               
               
        imm_beta_df["Peptide Length"] = imm_beta_df["Peptide"].str.len()
        imm_beta_df = imm_beta_df[(imm_beta_df["Peptide Length"]>=MIN_PEPTIDE_LEN)&(imm_beta_df["Peptide Length"]<=MAX_PEPTIDE_LEN)]
        
        imm_beta_df = imm_beta_df.sort_values(by="Number of Subjects Tested", 
                                              ascending=False).drop_duplicates(
                                                  "Sample", keep="first").reset_index(drop=True)
                                              
        min_subjects_tested = config.get("min_subjects_tested", 1)
        imm_beta_df = imm_beta_df[imm_beta_df["Number of Subjects Tested"]>=min_subjects_tested]
        
        
        imm_beta_df = imm_beta_df.dropna()
        
        imm_beta_df = filter_peptide_dataset(imm_beta_df, sorted_vocabulary)
        hla_filter = config.get("hla_filter", None)
        if hla_filter is not None:
            imm_beta_df = imm_beta_df[imm_beta_df["HLA"].str.startswith(hla_filter)]
        imm_beta_df = imm_beta_df.reset_index(drop=True)
            
        if config.get("beta_prior", None) is not None:
            selected_priors = beta_priors[config["beta_prior"]]
            
            print("Applying chosen prior..")
            for ix in tqdm(range(len(imm_beta_df))):
                row = imm_beta_df.iloc[ix]
                add_a, add_b = selected_priors[row["Qualitative Measurement"]]
                imm_beta_df.at[ix, "Alpha"] += add_a
                imm_beta_df.at[ix, "Beta"] += add_b
                imm_beta_df.at[ix, "Distr. Mean"] = beta_distr_mean(imm_beta_df.at[ix, "Alpha"], imm_beta_df.at[ix, "Beta"])
                imm_beta_df.at[ix, "Distr. Variance"] = beta_distr_var(imm_beta_df.at[ix, "Alpha"], imm_beta_df.at[ix, "Beta"])
                
        a = imm_beta_df["Number of Subjects Positive"] + 1
        b = imm_beta_df["Number of Subjects Tested"] - imm_beta_df["Number of Subjects Positive"] + 1
        imm_beta_df["Obs. Mean"] = a/(a+b)
        imm_beta_df["Obs. Variance"] = a*b/((a+b)**2 * (a+b+1))
        
        mean_threshold = config.get("mean_threshold", None)
        if mean_threshold is not None and mean_threshold!="none":
            n_starting_samples = len(imm_beta_df)
            pos_imm_beta_df = imm_beta_df[(imm_beta_df["Target"]==1)&(imm_beta_df["Distr. Mean"]>mean_threshold)]
            neg_imm_beta_df = imm_beta_df[(imm_beta_df["Target"]==0)&(imm_beta_df["Distr. Mean"]<=mean_threshold)]
            imm_beta_df = pd.concat([pos_imm_beta_df, neg_imm_beta_df])
            imm_beta_df = imm_beta_df.sample(frac=1, replace=False)
            n_filtered_samples = n_starting_samples - len(imm_beta_df)
            print(f"Filtered {n_filtered_samples} samples based on threshold")
            
        imm_beta_df["Stratification_index"] = imm_beta_df["HLA"] + "_" + imm_beta_df["Target"].astype(str)
        
        ix = imm_beta_df["Stratification_index"].value_counts()
        low_count_labels = ix[ix<3].index
        
        res_df = imm_beta_df[imm_beta_df["Stratification_index"].isin(low_count_labels)]
        imm_beta_df = imm_beta_df[~imm_beta_df["Stratification_index"].isin(low_count_labels)]

        test_size = config.get("test_size", 0.15)
        trainval_df, test_df = train_test_split(imm_beta_df, test_size=test_size, stratify=imm_beta_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    val_size = config.get("val_size", 0.1)
    train_df, val_df = train_test_split(trainval_df, test_size=val_size, stratify=trainval_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    if res_df is not None:
        train_df = pd.concat([train_df, res_df])

        
    dhlap_imm_df = pd.read_csv(config['dhlap_df'])
    dhlap_imm_df["Allele Pseudo-sequence"] = dhlap_imm_df["HLA"].str.replace("*", "", regex=False).map(hla_mapping)
    dhlap_imm_df = dhlap_imm_df.dropna()
    
    # Filter DHLAP to keep only samples not trained on  
    iedb_samples = set(tuple(x) for x in trainval_df[["Peptide", "HLA"]].values).union(set(tuple(x) for x in res_df[["Peptide", "HLA"]].values))
    dhlap_samples = set(tuple(x) for x in dhlap_imm_df[["Peptide", "HLA"]].values)
    dhlap_imm_df = dhlap_imm_df[dhlap_imm_df[["Peptide", "HLA"]].apply(tuple, 1).isin(dhlap_samples.difference(iedb_samples))]
    
    # dhlap_imm_df = dhlap_imm_df.merge(blast_df, left_on="Peptide", right_on="Peptide")
    dhlap_imm_df = dhlap_imm_df.dropna()
    dhlap_imm_df = filter_peptide_dataset(dhlap_imm_df, amino_acids)
    if dhlap_imm_df is not None and hla_filter is not None:
        dhlap_imm_df = dhlap_imm_df[dhlap_imm_df["HLA"].str.startswith(hla_filter)]
    dhlap_imm_df = dhlap_imm_df.reset_index(drop=True)

    if split_data:
        print("IEDB N. training samples: {}".format(len(train_df)))
        print("IEDB N. val samples: {}".format(len(val_df)))
        print("IEDB N. test samples: {}".format(len(test_df)))
        
        return train_df, val_df, test_df, dhlap_imm_df
    else:
        return imm_beta_df, dhlap_imm_df
    
    
def load_immunogenicity_dataframes_jointseqs(config, split_data=True):
    ps_df = pd.read_csv(config['pseudo_seq_file'])
    prot_df = pd.read_csv(config['hla_prot_seq_file'])


    hla_pseq_mapping = dict(ps_df[["HLA", "sequence"]].values)
    hla_prot_mapping = dict(prot_df[["HLA", "sequence"]].values)

    # ps_df = pd.read_csv(config['pseudo_seq_file'], sep="\t")
    # hla_mapping = dict(ps_df[["HLA", "sequence"]].values)

    res_df = None

    imm_df = pd.read_csv(config['immunogenicity_df'])
    imm_df["Allele Pseudo-sequence"] = imm_df["HLA"].str.replace("*", "", regex=False).map(hla_pseq_mapping)
    imm_df["Allele Protein sequence"] = imm_df["HLA"].str.replace("*", "", regex=False).map(hla_prot_mapping)

    imm_df = imm_df.dropna(subset=["Peptide", "Allele Pseudo-sequence"])
    imm_df["Target"] = (imm_df["Qualitative Measurement"]!="Negative").astype(int).values
    imm_df["Sample"] = imm_df["Peptide"] + "_" + imm_df["HLA"]


    imm_df["Peptide Length"] = imm_df["Peptide"].str.len()
    imm_df = imm_df[(imm_df["Peptide Length"]>=MIN_PEPTIDE_LEN)&(imm_df["Peptide Length"]<=MAX_PEPTIDE_LEN)]

    imm_df = imm_df.sort_values(by="Number of Subjects Tested", 
                                          ascending=False).drop_duplicates(
                                              "Sample", keep="first").reset_index(drop=True)

    min_subjects_tested = config.get("min_subjects_tested", 1)
    imm_df = imm_df[imm_df["Number of Subjects Tested"]>=min_subjects_tested]


    imm_df = imm_df.dropna()

    imm_df = filter_peptide_dataset(imm_df, sorted_vocabulary)
    hla_filter = config.get("hla_filter", None)
    if hla_filter is not None:
        imm_df = imm_df[imm_df["HLA"].str.startswith(hla_filter)]
    imm_df = imm_df.reset_index(drop=True)

    a = imm_df["Alpha"]
    b = imm_df["Beta"]
    imm_df["Distr. Mean"] = a/(a+b)
    imm_df["Distr. Variance"] = a*b/((a+b)**2 * (a+b+1))
    imm_df["Distr. Mode"] = (a-1)/(a+b-2)
    imm_df["Distr. Precision"] = a+b

    imm_df["Stratification_index"] = imm_df["HLA"] + "_" + imm_df["Target"].astype(str)

    ix = imm_df["Stratification_index"].value_counts()
    low_count_labels = ix[ix<3].index

    res_df = imm_df[imm_df["Stratification_index"].isin(low_count_labels)]
    imm_df = imm_df[~imm_df["Stratification_index"].isin(low_count_labels)]

    test_size = config.get("test_size", 0.15)
    trainval_df, test_df = train_test_split(imm_df, test_size=test_size, stratify=imm_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    val_size = config.get("val_size", 0.1)
    train_df, val_df = train_test_split(trainval_df, test_size=val_size, stratify=trainval_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    if res_df is not None:
        train_df = pd.concat([train_df, res_df])


    dhlap_imm_df = pd.read_csv(config['dhlap_df'])
    dhlap_imm_df["Allele Pseudo-sequence"] = dhlap_imm_df["HLA"].str.replace("*", "", regex=False).map(hla_pseq_mapping)
    dhlap_imm_df["Allele Protein sequence"] = dhlap_imm_df["HLA"].str.replace("*", "", regex=False).map(hla_prot_mapping)
    dhlap_imm_df = dhlap_imm_df.dropna()

    # Filter DHLAP to keep only samples not trained on  
    iedb_samples = set(tuple(x) for x in trainval_df[["Peptide", "HLA"]].values).union(set(tuple(x) for x in res_df[["Peptide", "HLA"]].values))
    dhlap_samples = set(tuple(x) for x in dhlap_imm_df[["Peptide", "HLA"]].values)
    dhlap_imm_df = dhlap_imm_df[dhlap_imm_df[["Peptide", "HLA"]].apply(tuple, 1).isin(dhlap_samples.difference(iedb_samples))]

    # dhlap_imm_df = dhlap_imm_df.merge(blast_df, left_on="Peptide", right_on="Peptide")
    dhlap_imm_df = dhlap_imm_df.dropna()
    dhlap_imm_df = filter_peptide_dataset(dhlap_imm_df, amino_acids)
    if dhlap_imm_df is not None and hla_filter is not None:
        dhlap_imm_df = dhlap_imm_df[dhlap_imm_df["HLA"].str.startswith(hla_filter)]
    dhlap_imm_df = dhlap_imm_df.reset_index(drop=True)

    if split_data:
        print("IEDB N. training samples: {}".format(len(train_df)))
        print("IEDB N. val samples: {}".format(len(val_df)))
        print("IEDB N. test samples: {}".format(len(test_df)))

        return train_df, val_df, test_df, dhlap_imm_df
    else:
        return imm_df, dhlap_imm_df



def load_immunogenicity_dataframes_calibration(config, split_data=True):
    ps_df = pd.read_csv(config['pseudo_seq_file'], sep="\t")
    hla_mapping = dict(ps_df[["HLA", "sequence"]].values)

    res_df = None

    imm_df = pd.read_csv(config['immunogenicity_df'])
    imm_df["Allele Pseudo-sequence"] = imm_df["HLA"].str.replace("*", "", regex=False).map(hla_mapping)
    imm_df = imm_df.dropna(subset=["Peptide", "Allele Pseudo-sequence"])
    imm_df["Target"] = (imm_df["Qualitative Measurement"]!="Negative").astype(int).values
    imm_df["Sample"] = imm_df["Peptide"] + "_" + imm_df["HLA"]
            
            
    imm_df["Peptide Length"] = imm_df["Peptide"].str.len()
    imm_df = imm_df[(imm_df["Peptide Length"]>=MIN_PEPTIDE_LEN)&(imm_df["Peptide Length"]<=MAX_PEPTIDE_LEN)]
    
    imm_df = imm_df.sort_values(by="Number of Subjects Tested", 
                                            ascending=False).drop_duplicates(
                                                "Sample", keep="first").reset_index(drop=True)
                                            
    min_subjects_tested = config.get("min_subjects_tested", 1)
    imm_df = imm_df[imm_df["Number of Subjects Tested"]>=min_subjects_tested]
    imm_df = imm_df.dropna()
    
    imm_df = filter_peptide_dataset(imm_df, sorted_vocabulary)
    hla_filter = config.get("hla_filter", None)
    if hla_filter is not None:
        imm_df = imm_df[imm_df["HLA"].str.startswith(hla_filter)]
    imm_df = imm_df.reset_index(drop=True)
        
    imm_df["Stratification_index"] = imm_df["HLA"] + "_" + imm_df["Target"].astype(str)
    
    ix = imm_df["Stratification_index"].value_counts()
    low_count_labels = ix[ix<3].index
    
    res_df = imm_df[imm_df["Stratification_index"].isin(low_count_labels)]
    imm_df = imm_df[~imm_df["Stratification_index"].isin(low_count_labels)]

    test_size = config.get("test_size", 0.15)
    calib_size = config.get("calib_size", 0.1)
    trainval_df, testcal_df = train_test_split(imm_df, test_size=test_size+calib_size, stratify=imm_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    val_size = config.get("val_size", 0.1)
    train_df, val_df = train_test_split(trainval_df, test_size=val_size, stratify=trainval_df["Stratification_index"], random_state=config['seed'], shuffle=True)
    
    test_frac = test_size/(test_size+calib_size)
    calib_df, test_df = train_test_split(testcal_df, test_size=test_frac, random_state=config['seed'], shuffle=True)
    
    if res_df is not None:
        train_df = pd.concat([train_df, res_df])

        
    dhlap_imm_df = pd.read_csv(config['dhlap_df'])
    dhlap_imm_df["Allele Pseudo-sequence"] = dhlap_imm_df["HLA"].str.replace("*", "", regex=False).map(hla_mapping)
    dhlap_imm_df = dhlap_imm_df.dropna()
    
    # Filter DHLAP to keep only samples not trained on  
    iedb_samples = set(tuple(x) for x in trainval_df[["Peptide", "HLA"]].values).union(set(tuple(x) for x in res_df[["Peptide", "HLA"]].values))
    dhlap_samples = set(tuple(x) for x in dhlap_imm_df[["Peptide", "HLA"]].values)
    dhlap_imm_df = dhlap_imm_df[dhlap_imm_df[["Peptide", "HLA"]].apply(tuple, 1).isin(dhlap_samples.difference(iedb_samples))]
    
    # dhlap_imm_df = dhlap_imm_df.merge(blast_df, left_on="Peptide", right_on="Peptide")
    dhlap_imm_df = dhlap_imm_df.dropna()
    dhlap_imm_df = filter_peptide_dataset(dhlap_imm_df, amino_acids)
    if dhlap_imm_df is not None and hla_filter is not None:
        dhlap_imm_df = dhlap_imm_df[dhlap_imm_df["HLA"].str.startswith(hla_filter)]
    dhlap_imm_df = dhlap_imm_df.reset_index(drop=True)

    if split_data:
        print("IEDB N. training samples: {}".format(len(train_df)))
        print("IEDB N. val samples: {}".format(len(val_df)))
        print("IEDB N. test samples: {}".format(len(test_df)))
        
        return train_df, val_df, calib_df, test_df, dhlap_imm_df
    else:
        return imm_df, dhlap_imm_df
    
    

class BetaDistributionDataset(Dataset):
    def __init__(self, df, hla_repr=["Allele Pseudo-sequence"]):
        super().__init__()
        cols = ["Peptide", *hla_repr, 
                "Alpha", "Beta", "Target"]
        self.data_matrix = df[cols].values.tolist()
        
    def __len__(self):
        return len(self.data_matrix)
    
    def __getitem__(self, ix):
        return self.data_matrix[ix]



class BetaDistributionDataset_ESM2Embs(Dataset):
    def __init__(self, df, peptide_embs, peptide2idx_mapping, hla_repr=["Allele Pseudo-sequence"]):
        super().__init__()
        cols = ["Peptide", *hla_repr, 
                "Alpha", "Beta", "Target"]
        self.data_matrix = df[cols].values.tolist()
        
        peptides = df["Peptide"].values
        idxs = [int(peptide2idx_mapping[p]) for p in peptides]
        # print(idxs)
        self.peptide_esm2_embeddings = [peptide_embs[i] for i in idxs]
        
        
    def __len__(self):
        return len(self.data_matrix)
    
    def __getitem__(self, ix):
        return self.data_matrix[ix], self.peptide_esm2_embeddings[ix]
