# Generated from Analysis-KCNQ2_harm-C-N-CA-restraints.ipynb.
# The working directory is set to this script's directory so ../ paths resolve to the repository root.
from pathlib import Path as _NotebookPath
import os as _notebook_os
_notebook_os.chdir(_NotebookPath(__file__).resolve().parent)

from pathlib import Path
import pandas as pd
from analysis_utils import combine_dataframes, ddg_barplot


# Load labels.
labels_data = pd.read_csv('../labels/kcnq2_labels.csv')
labels = labels_data.set_index('ligand')['label'].to_dict()
# Find per-MD data.
results_directory = Path('../results/kcnq2_harm-c-n-ca-restraints')
files_a = sorted(results_directory.glob('kcnq2-a_harm-c-n-ca-restraints_md*.csv'))
files_i = sorted(results_directory.glob('kcnq2-i_harm-c-n-ca-restraints_md*.csv'))
# Read CSV files into dataframes.
df_a = pd.concat([pd.read_csv(file) for file in files_a], ignore_index=True)
df_i = pd.concat([pd.read_csv(file) for file in files_i], ignore_index=True)
# Combine active and inactive data.
df = combine_dataframes(df_a, df_i)
# Sort the data.
df = df.sort_values('DDG')[['Ligand_Name', 'DGA', 'DGI', 'DDG', 'DGA_Error', 'DGI_Error', 'DDG_Error']]


_,_ = ddg_barplot(
    df, labels, figsize=(4, 2.5), ylim=[-6.5, 5.5], errorbars=False,
    file_name='../plots/kcnq2_ddg_barplot_harm-c-n-ca-restraints'
)
