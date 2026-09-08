# Generated from Analysis-a3b4-nAChR_short-repeats.ipynb.
# The working directory is set to this script's directory so ../ paths resolve to the repository root.
from pathlib import Path as _NotebookPath
import os as _notebook_os
_notebook_os.chdir(_NotebookPath(__file__).resolve().parent)

try:
    from IPython.display import display
except ImportError:
    display = print

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from analysis_utils import combine_dataframes


# Discover per-seed standardized CSVs.
results_directory = Path('../results/a3b4-nachr_short-repeats')
files_a = sorted(results_directory.glob('a3b4-nachr-a-short_seed-*.csv'))
files_i = sorted(results_directory.glob('a3b4-nachr-i-short_seed-*.csv'))
# Ccheck that the number of files for active and inactive seeds match.
if len(files_a) != len(files_i):
    raise ValueError('Active and inactive seed file counts must match.')
# Load combined data from per-seed CSVs.
seed = list(range(len(files_a)))
combined_a = pd.concat([pd.read_csv(file) for file in files_a], ignore_index=True)
combined_i = pd.concat([pd.read_csv(file) for file in files_i], ignore_index=True)
# Remove ligand 'dummy' if they exist.
combined_a = combined_a[combined_a['ligand_name'] != 'dummy']
combined_i = combined_i[combined_i['ligand_name'] != 'dummy']


# Group combined data by ligand name and calculate statistics.
grouped_a = combined_a.groupby('ligand_name')['pred_dg'].agg(
    Average=('mean'),
    StdDev=('std'),
    StdErr=('sem'),
)
grouped_i = combined_i.groupby('ligand_name')['pred_dg'].agg(
    Average=('mean'),
    StdDev=('std'),
    StdErr=('sem'),
)


# Merge grouped active and inactive dataframes and calculate DDG.
merged_df = pd.merge(
    grouped_a,
    grouped_i,
    on='ligand_name',
    how='outer',
    suffixes=('_a', '_i'),
).reset_index()
merged_df['DDG'] = merged_df['Average_a'] - merged_df['Average_i']
merged_df = merged_df.sort_values(by='DDG', ascending=False, ignore_index=True)
# Load labels
labels_data = pd.read_csv('../labels/a3b4-nachr_labels.csv')
labels = labels_data.set_index('ligand')['label'].to_dict()
merged_df['label'] = merged_df['ligand_name'].map(labels)
display(merged_df)


fig, ax = plt.subplots(1, 1, figsize=[4, 3.5], dpi=300)

# --- Generate the Bar Plot ---
avg_cols = ['Average_i', 'Average_a']
std_cols = ['StdDev_i', 'StdDev_a']
err_cols = ['StdErr_i', 'StdErr_a']
offset = [-0.2, 0.2]
label = ['inactive', 'active']
color = ['C0', 'C1']
for i in range(2):
    ax.bar(np.arange(len(merged_df['ligand_name'][1:6])) + offset[i],
           merged_df[avg_cols[i]][1:6],
           yerr=merged_df[std_cols[i]][1:6],
           capsize=4,
           width=0.4,
           label=label[i],
           color=color[i],
           alpha=0.5,
           ecolor=color[i]
    )
    ax.bar(np.arange(len(merged_df['ligand_name'][1:6])) + offset[i],
           merged_df[avg_cols[i]][1:6],
           yerr=merged_df[err_cols[i]][1:6],
           capsize=2,
           width=0.4,
           alpha=0.0,
           ecolor=color[i]
    )

ax.set_xticks(range(len(merged_df['ligand_name'][1:6])))
ax.set_xticklabels(merged_df['ligand_name'][1:6])
ax.set_ylabel(r'$\Delta G$ [kcal/mol]', size=12)
ax.set_xlim(-0.6, 4.6)
ax.set_ylim(-23, -7)
for tick_label in ax.get_xticklabels():
    ligand_name = tick_label.get_text()
    if merged_df.loc[merged_df['ligand_name'] == ligand_name, 'label'].values[0] == 'agonist':
        tick_label.set_color('C1')
    elif merged_df.loc[merged_df['ligand_name'] == ligand_name, 'label'].values[0] == 'antagonist':
        tick_label.set_color('C0')
ax.set_xticklabels(merged_df['ligand_name'][1:6], rotation=90)
ax.legend(fontsize=12)
fig.tight_layout()
fig.savefig('../plots/a3b4-nAChR_short-repeats_barplot.png')
fig.savefig('../plots/a3b4-nAChR_short-repeats_barplot.pdf')
plt.show()
