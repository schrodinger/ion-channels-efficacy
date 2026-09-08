# Generated from Analysis-GABAA-rho1.ipynb.
# The working directory is set to this script's directory so ../ paths resolve to the repository root.
from pathlib import Path as _NotebookPath
import os as _notebook_os
_notebook_os.chdir(_NotebookPath(__file__).resolve().parent)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_utils import combine_dataframes, ddg_barplot


labels_data = pd.read_csv('../labels/gabaa-rho1_labels.csv')
labels = labels_data.set_index('ligand')['label'].to_dict()
names = labels_data.set_index('ligand')['name'].to_dict()


# Load PDB-specific data before creating state-specific groupings.
pdb_ids = ['8OP9', '8RH7', '8OQ7', '9FRB']
data_by_pdb = {
    pdb_id: pd.read_csv(f'../results/gabaa-rho1-{pdb_id.lower()}.csv')
    for pdb_id in pdb_ids
}

df_p = data_by_pdb['8RH7']
df_d = data_by_pdb['8OP9']
df_8oq7 = data_by_pdb['8OQ7']
df_9frb = data_by_pdb['9FRB']

combined_a = pd.concat([df_d, df_p], ignore_index=True)
combined_i = pd.concat([df_8oq7, df_9frb], ignore_index=True)


df_active_inactive = combine_dataframes(combined_a, combined_i)
df_primed_inactive = combine_dataframes(df_p, combined_i)
df_desensitized_inactive = combine_dataframes(df_d, combined_i)
# Sort each dataframe by DDG
df_active_inactive = df_active_inactive.sort_values(by='DDG')
df_primed_inactive = df_primed_inactive.sort_values(by='DDG')
df_desensitized_inactive = df_desensitized_inactive.sort_values(by='DDG')


_,_ = ddg_barplot(
    df_active_inactive, labels,
    xlabel_mode='dict', ligand_names=names,
    file_name='../plots/gabaa-rho1_barplot'
    )


_,_ = ddg_barplot(
    df_active_inactive, labels,
    xlabel_mode='none', ligand_names=names,
    file_name='../plots/gabaa-rho1_barplot_no-labels.png'
    )


_,_ = ddg_barplot(
    df_primed_inactive, labels, figsize=(4, 4),
    xlabel_mode='dict', ligand_names=names,
    file_name='../plots/gabaa-rho1_barplot_primed_inactive'
    )
_,_ = ddg_barplot(
    df_desensitized_inactive, labels, figsize=(4, 4),
    xlabel_mode='dict', ligand_names=names,
    file_name='../plots/gabaa-rho1_barplot_desensitized_inactive'
    )
