import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

import os
import pandas as pd
import numpy as np

import const
import utils

drug_name_2_index = { v: index[0] for index, v in np.ndenumerate(pd.read_csv(const.drugs_out_file)['Drug'].to_numpy()) }
drug_lookup = pd.read_csv(const.drugs_out_file)['Drug'].to_numpy()
diagnosis_name_2_index = { v: index[0] for index, v in np.ndenumerate(pd.read_csv(const.diagnosis_out_file)['Diagnosis'].to_numpy()) }
diagnosis_lookup = pd.read_csv(const.diagnosis_out_file)['Diagnosis'].to_numpy()
biochemistry_lookup = pd.read_csv(const.biochemistry_out_file)['Biochemistry'].to_numpy()

class PharmacistDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        df = pd.read_csv(annotations_file)

        # IPs
        self.IPs = df['IP'].to_numpy()

        # labels
        drugs = [ np.array(strLabels.split(';')) for strLabels in df["Labels"].to_numpy() ]
        labels = np.full((len(drugs), len(drug_name_2_index)), 0.0, dtype=float)
        for r, drug_array in np.ndenumerate(drugs):
            for drug in drug_array:
                labels[r][drug_name_2_index[drug]] = 1.0

        # diagnosis
        features = np.full((len(labels), 44 + len(diagnosis_name_2_index)), 0.0, dtype=float)

        diagnosis = [ np.array(strDiagnosis.split(';')) for strDiagnosis in df["所有诊断与编码"].to_numpy() ]
        for r, diagnosis_array in np.ndenumerate(diagnosis):
            for diagnosis in diagnosis_array:
                features[r][diagnosis_name_2_index[diagnosis]] = 1.0

        # biochemistry
        biochemistry = df.iloc[:, 2:46].to_numpy()
        print(f'biochemistry: {biochemistry.shape}')
        for r in range(len(biochemistry)):
            for c in range(44):
                features[r][c + len(diagnosis_name_2_index)] = biochemistry[r][c]

        print(features.shape)

        self.features = torch.from_numpy(features)
        self.labels = torch.Tensor(labels)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feature = self.features[idx]
        label = self.labels[idx]
        IP = self.IPs[idx]

        if self.transform:
            feature = self.transform(feature)
        if self.target_transform:
            label = self.target_transform(label)
        return feature, label, IP

# MARK: - check whether dataset is right

def printPharmacistDataset():
    dataset = PharmacistDataset(const.labels_features_output_file)
    for i in range(3357):
        feature = dataset[i][0]
        out_feature = []
        for f in range(len(diagnosis_name_2_index)):
            if feature[f] == 1:
                out_feature.append(diagnosis_lookup[f]) 
        for f in range(44):
            if feature[f + len(diagnosis_name_2_index)] == 1:
                out_feature.append(biochemistry_lookup[f])

        label = dataset[i][1]
        out_label = []
        for l in range(len(drug_name_2_index)):
            if label[l] == 1:
                out_label.append(drug_lookup[l])

        print(f'🌶️🌶️🌶️🌶️🌶️🌶️🌶️🌶️🌶️🌶️🌶️\n{i}-{dataset[i][2]}\nfeature: {out_feature}\nlabels: {out_label}\n🫑🫑🫑🫑🫑🫑🫑🫑🫑🫑🫑')

# printPharmacistDataset()